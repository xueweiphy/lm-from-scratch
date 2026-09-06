"""Byte-Pair Encoding (BPE) tokenizer.

The tokenizer works at the byte level: text is first split into
"pretokens" with the GPT-2 regex pattern, each pretoken is turned into a
sequence of UTF-8 byte IDs (0-255), and training repeatedly merges the
most frequent adjacent pair of token IDs into a new token until the
vocabulary reaches the requested size.  Special tokens (e.g.
"<|endoftext|>") are never split or merged across.

Example
-------
    from bpe import BPE

    tokenizer = BPE(special_tokens=["<|endoftext|>"])
    vocab, merges = tokenizer.train(text, vocab_size=500)

    ids = tokenizer.encode("Hello world!<|endoftext|>")
    text_back = tokenizer.decode(ids)
"""

import json

import regex as re

# GPT-2 pretokenization pattern
PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


class BPE:
    """Byte-level BPE tokenizer with training, encoding and decoding."""

    def __init__(self, special_tokens=None):
        self.special_tokens = list(special_tokens) if special_tokens else []
        # vocab: token_id -> bytes.  IDs 0-255 are the single bytes.
        self.vocab = {token_id: bytes([token_id]) for token_id in range(256)}
        # merges: list of (left_id, right_id) pairs, in the order learned
        self.merges = []
        # merge_to_token_id: (left_id, right_id) -> new token id
        self.merge_to_token_id = {}
        # special token string -> token id (assigned after training)
        self.special_token_ids = {}

    # ------------------------------------------------------------------
    # Pretokenization
    # ------------------------------------------------------------------
    @staticmethod
    def position_special_tokens(data_input, special_tokens):
        """Find start/end positions of every special-token occurrence."""
        pos_start = []
        pos_end = []
        for token in special_tokens:
            start = 0
            position = data_input.find(token, start)
            while True:
                if position == -1:
                    break
                pos_start += [position]
                pos_end += [position + len(token)]
                start = position + len(token)
                position = data_input.find(token, start)

        pos_start = sorted(pos_start)
        pos_end = sorted(pos_end)
        return [pos_start, pos_end]

    def split_on_special_tokens(self, data_input):
        """Split text into chunks that contain no special tokens."""
        if not self.special_tokens:
            return [data_input]
        pos_start, pos_end = self.position_special_tokens(data_input, self.special_tokens)
        pos_start = pos_start + [None]
        pos_end = [0] + pos_end
        return [data_input[ii:jj] for ii, jj in zip(pos_end, pos_start)]

    def pretokenize(self, data_input):
        """Split text into pretoken strings (special tokens removed)."""
        str_list = []
        for chunk in self.split_on_special_tokens(data_input):
            str_list += re.findall(PAT, chunk)
        return str_list

    # ------------------------------------------------------------------
    # Counting
    # ------------------------------------------------------------------
    @staticmethod
    def count_pretokens(pretoken_strings):
        """Count unique pretokens and map them to byte-ID sequences."""
        pretoken_text_counts = {}
        pretoken_counts = {}
        pretoken_token_ids = {}
        for pretoken_text in pretoken_strings:
            pretoken_text_counts[pretoken_text] = pretoken_text_counts.get(pretoken_text, 0) + 1

        pretoken_id = 0
        for pretoken_text, frequency in pretoken_text_counts.items():
            pretoken_counts[pretoken_id] = frequency
            pretoken_token_ids[pretoken_id] = list(pretoken_text.encode("utf-8"))
            pretoken_id += 1
        return pretoken_counts, pretoken_token_ids

    @staticmethod
    def count_pairs(pretoken_counts, pretoken_token_ids):
        """Count adjacent pairs, weighted by pretoken frequencies."""
        pair_counts = {}
        pair_to_pretoken_ids = {}
        for pretoken_id, frequency in pretoken_counts.items():
            token_ids = pretoken_token_ids[pretoken_id]
            for index, left_token_id in enumerate(token_ids[:-1]):
                pair = (left_token_id, token_ids[index + 1])
                pair_counts[pair] = pair_counts.get(pair, 0) + frequency
                pair_to_pretoken_ids.setdefault(pair, set()).add(pretoken_id)
        return pair_counts, pair_to_pretoken_ids

    # ------------------------------------------------------------------
    # Merging
    # ------------------------------------------------------------------
    @staticmethod
    def merge_pair(selected_pair, new_token_id, pair_to_pretoken_ids,
                   pretoken_token_ids, pretoken_counts, pair_counts):
        """Merge a selected pair everywhere and update affected pair counts.

        Only the pretokens that contain the selected pair are touched: for
        each one, its old pair contributions are subtracted and the new
        ones (after the merge) are added back.  This stays incremental but
        also handles overlapping occurrences (e.g. "ababab") correctly.
        """
        affected_pretoken_ids = pair_to_pretoken_ids[selected_pair]
        for pretoken_id in affected_pretoken_ids:
            old_token_ids = pretoken_token_ids[pretoken_id]
            frequency = pretoken_counts[pretoken_id]

            # Subtract this pretoken's old pair contributions
            for index, left_token_id in enumerate(old_token_ids[:-1]):
                old_pair = (left_token_id, old_token_ids[index + 1])
                pair_counts[old_pair] -= frequency

            # Merge every occurrence of the selected pair
            index = 0
            merged_token_ids = []
            while index < len(old_token_ids):
                if index < len(old_token_ids) - 1 and \
                        selected_pair == (old_token_ids[index], old_token_ids[index + 1]):
                    merged_token_ids += [new_token_id]
                    index += 2
                else:
                    merged_token_ids += [old_token_ids[index]]
                    index += 1
            pretoken_token_ids[pretoken_id] = merged_token_ids

            # Add this pretoken's new pair contributions
            for index, left_token_id in enumerate(merged_token_ids[:-1]):
                new_pair = (left_token_id, merged_token_ids[index + 1])
                pair_counts[new_pair] = pair_counts.get(new_pair, 0) + frequency
                pair_to_pretoken_ids.setdefault(new_pair, set()).add(pretoken_id)

        # Drop pairs whose count fell to zero (including the merged pair)
        for pair in [pair for pair, count in pair_counts.items() if count <= 0]:
            pair_counts.pop(pair)
        pair_counts.pop(selected_pair, None)
        return pretoken_token_ids

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    def train(self, data_input, vocab_size, verbose=False):
        """Learn BPE merges from text until the vocab reaches vocab_size.

        vocab_size counts the 256 byte tokens, the learned merges, and
        the special tokens (appended at the end).
        """
        num_merges = vocab_size - 256 - len(self.special_tokens)
        if num_merges < 0:
            raise ValueError("vocab_size must be at least 256 + number of special tokens")

        pretoken_strings = self.pretokenize(data_input)
        pretoken_counts, pretoken_token_ids = self.count_pretokens(pretoken_strings)
        pair_counts, pair_to_pretoken_ids = self.count_pairs(pretoken_counts, pretoken_token_ids)

        self.vocab = {token_id: bytes([token_id]) for token_id in range(256)}
        self.merges = []
        self.merge_to_token_id = {}

        next_token_id = 256
        while next_token_id < 256 + num_merges:
            if not pair_counts or max(pair_counts.values()) <= 0:
                break  # nothing left to merge
            # Most frequent pair; ties broken by the lexicographically
            # greater pair of *byte strings* (CS336 spec), not of IDs
            selected_pair = max(
                pair_counts,
                key=lambda candidate_pair: (
                    pair_counts[candidate_pair],
                    (self.vocab[candidate_pair[0]], self.vocab[candidate_pair[1]]),
                ),
            )
            if verbose:
                print(f"merge {next_token_id - 256}: {selected_pair} -> {next_token_id} "
                      f"(count {pair_counts[selected_pair]})")
            self.merge_to_token_id[selected_pair] = next_token_id
            self.merges += [selected_pair]
            # Keep the vocab current: the byte-string tie-break above
            # needs every ID's bytes to be available as soon as it exists
            self.vocab[next_token_id] = (
                self.vocab[selected_pair[0]] + self.vocab[selected_pair[1]])
            self.merge_pair(selected_pair, next_token_id, pair_to_pretoken_ids,
                            pretoken_token_ids, pretoken_counts, pair_counts)
            next_token_id += 1

        # Append special tokens at the end of the vocab
        self.special_token_ids = {}
        for token in self.special_tokens:
            self.vocab[next_token_id] = token.encode("utf-8")
            self.special_token_ids[token] = next_token_id
            next_token_id += 1

        return self.vocab, self.merges

    # ------------------------------------------------------------------
    # Encoding / decoding
    # ------------------------------------------------------------------
    def encode_pretoken(self, tid):
        """Apply the learned merges, in order, to one list of token IDs."""
        tid = tid.copy()
        for merge_id, mm in enumerate(self.merges, start=256):
            jj = 0
            while jj < len(tid) - 1:
                if (tid[jj], tid[jj + 1]) == mm:
                    tid[jj:jj + 2] = [merge_id]
                    jj += 1
                else:
                    jj += 1
        return tid

    def encode(self, data_input):
        """Encode text into a list of token IDs.

        Each *unique* pretoken is merged only once (cached), then its IDs
        are emitted once per occurrence, in order.  Special tokens map to
        their own IDs from training.
        """
        out = []
        cache = {}
        if self.special_tokens:
            pos_start, pos_end = self.position_special_tokens(data_input, self.special_tokens)
        else:
            pos_start, pos_end = [], []
        specials = [data_input[ii:jj] for ii, jj in zip(pos_start, pos_end)]

        chunk_index = 0
        for ii, jj in zip([0] + pos_end, pos_start + [None]):
            for pretoken_text in re.findall(PAT, data_input[ii:jj]):
                if pretoken_text not in cache:
                    cache[pretoken_text] = self.encode_pretoken(list(pretoken_text.encode("utf-8")))
                out += cache[pretoken_text]
            if chunk_index < len(specials):
                out += [self.special_token_ids[specials[chunk_index]]]
            chunk_index += 1
        return out

    def decode(self, ids):
        """Decode a list of token IDs back into text."""
        strs_bytes = b''.join(self.vocab[dd] for dd in ids)
        strs = strs_bytes.decode("utf-8", errors="replace")
        return strs

    # ------------------------------------------------------------------
    # Save / load
    # ------------------------------------------------------------------
    def save(self, filepath):
        """Save vocab, merges and special tokens to a JSON file."""
        state = {
            "special_tokens": self.special_tokens,
            "merges": [list(pair) for pair in self.merges],
            "vocab": {str(token_id): token_bytes.hex()
                      for token_id, token_bytes in self.vocab.items()},
        }
        with open(filepath, "w") as file:
            json.dump(state, file, indent=2)

    @classmethod
    def load(cls, filepath):
        """Load a tokenizer previously saved with .save()."""
        with open(filepath) as file:
            state = json.load(file)
        tokenizer = cls(special_tokens=state["special_tokens"])
        tokenizer.merges = [tuple(pair) for pair in state["merges"]]
        tokenizer.vocab = {int(token_id): bytes.fromhex(hex_string)
                           for token_id, hex_string in state["vocab"].items()}
        next_token_id = 256
        tokenizer.merge_to_token_id = {}
        for pair in tokenizer.merges:
            tokenizer.merge_to_token_id[pair] = next_token_id
            next_token_id += 1
        tokenizer.special_token_ids = {}
        for token in tokenizer.special_tokens:
            tokenizer.special_token_ids[token] = next_token_id
            next_token_id += 1
        return tokenizer


if __name__ == "__main__":
    sample_text = (
        "low low low low low lower lower widest widest widest "
        "newest newest newest newest newest newest<|endoftext|>"
        "the quick brown fox jumps over the lazy dog"
    )

    tokenizer = BPE(special_tokens=["<|endoftext|>"])
    vocab, merges = tokenizer.train(sample_text, vocab_size=280, verbose=True)

    print("\nfirst merges:", merges[:10])
    print("new tokens:", {token_id: vocab[token_id] for token_id in range(256, 266)})

    encoded = tokenizer.encode("the newest lower dog<|endoftext|>")
    print("\nencoded:", encoded)
    print("decoded:", tokenizer.decode(encoded))
