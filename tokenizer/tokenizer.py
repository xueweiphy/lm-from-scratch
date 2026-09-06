"""BPE tokenizer for inference (assignment 1, `tokenizer`).

Training lives in bpe_multiprocessing.BPE.  This class is the other half:
it is constructed from an *already trained* vocabulary and merge list, and
turns text into token IDs and back.
"""

import json

import regex as re

from .bpe_multiprocessing import PAT


class Tokenizer:
    """Encode text to token IDs and back, given a vocab and merge list."""

    def __init__(self, vocab, merges, special_tokens=None):
        """
        vocab:          dict[int, bytes]
        merges:         list[tuple[bytes, bytes]], ordered by creation
        special_tokens: list[str] | None
        """
        self.vocab = dict(vocab)
        self.special_tokens = list(special_tokens) if special_tokens else []

        # bytes -> id, so merges (which are byte pairs) can be resolved to ids.
        self.token_ids = {token_bytes: token_id
                          for token_id, token_bytes in self.vocab.items()}

        # Append any special token that is not already in the vocabulary.
        next_id = max(self.vocab) + 1 if self.vocab else 0
        for special_token in self.special_tokens:
            token_bytes = special_token.encode("utf-8")
            if token_bytes not in self.token_ids:
                self.vocab[next_id] = token_bytes
                self.token_ids[token_bytes] = next_id
                next_id += 1
        self.special_token_ids = {token: self.token_ids[token.encode("utf-8")]
                                  for token in self.special_tokens}

        # merge_rank: (left_id, right_id) -> position in the merge list.
        # merge_result: (left_id, right_id) -> id of the merged token.
        # Ranks come from the merge order, never from the ids themselves --
        # a supplied vocabulary need not number merges 256, 257, ...
        self.merge_rank = {}
        self.merge_result = {}
        for rank, (left_bytes, right_bytes) in enumerate(merges):
            left_id = self.token_ids.get(left_bytes)
            right_id = self.token_ids.get(right_bytes)
            merged_id = self.token_ids.get(left_bytes + right_bytes)
            if left_id is None or right_id is None or merged_id is None:
                continue
            pair = (left_id, right_id)
            if pair not in self.merge_rank:
                self.merge_rank[pair] = rank
                self.merge_result[pair] = merged_id

        # Longest first, so "<|endoftext|><|endoftext|>" wins over "<|endoftext|>".
        self._sorted_specials = sorted(self.special_tokens, key=len, reverse=True)
        self._special_pattern = (
            "(" + "|".join(re.escape(t) for t in self._sorted_specials) + ")"
            if self._sorted_specials else None)
        self._max_special_len = max((len(t) for t in self.special_tokens), default=0)

        self._cache = {}

    @classmethod
    def from_files(cls, vocab_filepath, merges_filepath, special_tokens=None):
        """Build a Tokenizer from files written by BPE.save-style output.

        vocab_filepath:  JSON, {token_id: hex-encoded token bytes}
        merges_filepath: one merge per line, two hex-encoded tokens separated
                         by a space, in order of creation.
        """
        with open(vocab_filepath) as vocab_file:
            raw_vocab = json.load(vocab_file)
        vocab = {int(token_id): bytes.fromhex(hex_string)
                 for token_id, hex_string in raw_vocab.items()}

        merges = []
        with open(merges_filepath) as merges_file:
            for line in merges_file:
                line = line.rstrip("\n")
                if not line:
                    continue
                left_hex, right_hex = line.split(" ")
                merges.append((bytes.fromhex(left_hex), bytes.fromhex(right_hex)))

        return cls(vocab, merges, special_tokens)

    def _split_on_specials(self, text):
        """Yield (chunk, is_special) pairs, longest special matched first."""
        if self._special_pattern is None:
            yield text, False
            return
        for part in re.split(self._special_pattern, text):
            if part:
                yield part, part in self.special_token_ids

    def _encode_pretoken(self, pretoken_text):
        """Merge one pretoken, always applying the lowest-rank pair available."""
        ids = [self.token_ids[bytes([b])] for b in pretoken_text.encode("utf-8")]
        while len(ids) > 1:
            best_rank = None
            best_index = None
            for index in range(len(ids) - 1):
                rank = self.merge_rank.get((ids[index], ids[index + 1]))
                if rank is not None and (best_rank is None or rank < best_rank):
                    best_rank = rank
                    best_index = index
            if best_index is None:
                break
            pair = (ids[best_index], ids[best_index + 1])
            ids[best_index:best_index + 2] = [self.merge_result[pair]]
        return ids

    def encode(self, text):
        """Encode text into a list of token IDs."""
        out = []
        for chunk, is_special in self._split_on_specials(text):
            if is_special:
                out.append(self.special_token_ids[chunk])
                continue
            for pretoken_text in re.findall(PAT, chunk):
                if pretoken_text not in self._cache:
                    self._cache[pretoken_text] = self._encode_pretoken(pretoken_text)
                out += self._cache[pretoken_text]
        return out

    def _safe_split(self, buffer):
        """Split `buffer` into (ready, held) so nothing tokenizable straddles.

        A pretoken or a special token can span two pieces of the input, so the
        tail of the buffer is held back until more text arrives.
        """
        held_length = self._max_special_len
        if held_length >= len(buffer):
            return "", buffer
        ready, held = (buffer, "") if held_length == 0 else \
            (buffer[:-held_length], buffer[-held_length:])

        # A whitespace run or word can still grow, so return the last pretoken.
        pretokens = re.findall(PAT, ready)
        if pretokens:
            last = pretokens[-1]
            ready, held = ready[:len(ready) - len(last)], last + held

        # Do not cut inside a special token that started before the split.
        for length in range(min(len(ready), self._max_special_len - 1), 0, -1):
            suffix = ready[-length:]
            if any(t.startswith(suffix) and len(t) > length
                   for t in self.special_tokens):
                ready, held = ready[:-length], suffix + held
                break

        return ready, held

    def encode_iterable(self, iterable, chunk_chars=1 << 16):
        """Lazily encode an iterable of strings (e.g. an open file handle)."""
        buffer = ""
        for piece in iterable:
            buffer += piece
            if len(buffer) >= chunk_chars:
                ready, buffer = self._safe_split(buffer)
                yield from self.encode(ready)
        yield from self.encode(buffer)

    def decode(self, ids):
        """Decode a list of token IDs back into text."""
        token_bytes = b"".join(self.vocab[i] for i in ids if i in self.vocab)
        return token_bytes.decode("utf-8", errors="replace")
