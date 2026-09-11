"""Byte-Pair Encoding (BPE) tokenizer — multiprocessing version (steps 1-3).

Same tokenizer as bpe.py, plus *parallel pretokenization* of a large
file: the file is chunked at <|endoftext|> boundaries
(find_chunk_boundaries, from the CS336 starter code), each worker
process pretokenizes one chunk (pretokenize_chunk), and the workers'
Counters are merged into one pretoken-frequency Counter
(pretokenize_file).  Chunking at document boundaries is safe because no
BPE merge ever crosses a special token.

Training entry points (all share one merge loop, train_from_counts):
    train(text, vocab_size)                     — single process, from a string
    train_from_file(path, vocab_size, n_procs)  — parallel pretokenization

Run the checks (train == train_from_file, plus pretokenization timings):

    python bpe_multiprocessing.py
"""

import json
import os
import time
from collections import Counter
from multiprocessing import Pool
from typing import BinaryIO

import regex as re

# GPT-2 pretokenization pattern
PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


# ----------------------------------------------------------------------
# Step 2: chunk the file at special-token boundaries
# ----------------------------------------------------------------------
def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    """Chunk the file into parts that can be counted independently.

    May return fewer chunks if the boundaries end up overlapping.
    Verbatim from the CS336 starter code (cs336_basics/pretokenization_example.py).
    """
    assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)
        while True:
            mini_chunk = file.read(mini_chunk_size)

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    return sorted(set(chunk_boundaries))


# ----------------------------------------------------------------------
# Step 3: worker + Pool
# ----------------------------------------------------------------------
def pretokenize_chunk(args):
    """Worker: pretokenize one byte range of a file, return a Counter.

    Top-level function so multiprocessing (spawn) can pickle it by name.
    The arguments are tiny (path + offsets); the worker opens the file
    and reads only its own slice.  It returns pretoken *frequencies*,
    never the occurrence list — a Counter is a few hundred KB where the
    occurrence list would be hundreds of MB.
    """
    path, start, end, special_tokens = args
    with open(path, "rb") as file:
        file.seek(start)
        chunk = file.read(end - start).decode("utf-8", errors="ignore")


    tokenizer = BPE(special_tokens=special_tokens)
    #return Counter(tokenizer.pretokenize(chunk))

    counter = Counter()
    for document in tokenizer.split_on_special_tokens(chunk):
        counter.update(re.findall(PAT, document))
    return counter



def pretokenize_file(path, special_tokens, num_processes=None):
    """Pretokenize a whole file in parallel; return one merged Counter."""
    if num_processes is None:
        num_processes = os.cpu_count()

    with open(path, "rb") as file:
        boundaries = find_chunk_boundaries(
            file, num_processes *8 , special_tokens[0].encode("utf-8"))

    tasks = [(path, start, end, special_tokens)
             for start, end in zip(boundaries[:-1], boundaries[1:])]
    with Pool(processes=num_processes) as pool:
        chunk_counters = pool.map(pretokenize_chunk, tasks)

    pretoken_text_counts = Counter()
    for chunk_counter in chunk_counters:
        pretoken_text_counts.update(chunk_counter)
    return pretoken_text_counts


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
        """Learn BPE merges from a text string (single process)."""
        pretoken_strings = self.pretokenize(data_input)
        return self.train_from_counts(Counter(pretoken_strings), vocab_size, verbose)

    def train_from_file(self, path, vocab_size, num_processes=None, verbose=False,
                        progress_every=0):
        """Learn BPE merges from a file, pretokenizing in parallel."""
        counts = pretokenize_file(path, self.special_tokens, num_processes)
        return self.train_from_counts(counts, vocab_size, verbose, progress_every)

    def train_from_counts(self, pretoken_text_counts, vocab_size, verbose=False,
                          progress_every=0):
        """Learn BPE merges from pretoken frequencies ({text: count}).

        vocab_size counts the 256 byte tokens, the learned merges, and
        the special tokens (appended at the end).
        """
        num_merges = vocab_size - 256 - len(self.special_tokens)
        if num_merges < 0:
            raise ValueError("vocab_size must be at least 256 + number of special tokens")

        pretoken_counts = {}
        pretoken_token_ids = {}
        pretoken_id = 0
        for pretoken_text, frequency in pretoken_text_counts.items():
            pretoken_counts[pretoken_id] = frequency
            pretoken_token_ids[pretoken_id] = list(pretoken_text.encode("utf-8"))
            pretoken_id += 1

        pair_counts, pair_to_pretoken_ids = self.count_pairs(pretoken_counts, pretoken_token_ids)

        self.vocab = {token_id: bytes([token_id]) for token_id in range(256)}
        self.merges = []
        self.merge_to_token_id = {}

        start_time = time.time()
        if progress_every:
            print(f"merge loop: {len(pretoken_counts)} unique pretokens, "
                  f"{len(pair_counts)} initial pairs, {num_merges} merges to do",
                  flush=True)

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

            merges_done = next_token_id - 256
            if progress_every and merges_done % progress_every == 0:
                elapsed = time.time() - start_time
                rate = merges_done / elapsed
                remaining = (num_merges - merges_done) / rate
                print(f"  {merges_done}/{num_merges} merges, "
                      f"{elapsed / 60:.1f} min elapsed, "
                      f"~{remaining / 60:.0f} min remaining, "
                      f"{len(pair_counts)} live pairs", flush=True)

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
        ids = ids.tolist() if hasattr(ids, "tolist") else ids
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
    # Step-3 check: parallel pretokenization must equal serial, then time it.
    # Edit these three settings, then run:  python bpe_multiprocessing.py
    path = "sample_50MB.txt"
    num_processes = os.cpu_count()   # or try 2, 4, 8 and compare
    special_tokens = ["<|endoftext|>"]

    # Serial oracle (single process, whole file)
    tokenizer = BPE(special_tokens=special_tokens)
    with open(path, encoding="utf-8", errors="ignore") as file:
        data = file.read()
    t0 = time.time()
    serial_counts = Counter(tokenizer.pretokenize(data))
    t_serial = time.time() - t0
    del data

    # Parallel version
    t0 = time.time()
    parallel_counts = pretokenize_file(path, special_tokens, num_processes)
    t_parallel = time.time() - t0

    print(f"serial:   {t_serial:.1f}s")
    print(f"parallel: {t_parallel:.1f}s with {num_processes} processes "
          f"(speedup {t_serial / t_parallel:.1f}x)")
    print(f"counts identical: {serial_counts == parallel_counts} "
          f"({sum(parallel_counts.values())} pretokens, {len(parallel_counts)} unique)")

    # Step-4 check: full training via both paths must give identical results
    vocab_size = 1000
    tok_a = BPE(special_tokens=special_tokens)
    tok_a.train_from_counts(serial_counts, vocab_size)
    tok_b = BPE(special_tokens=special_tokens)
    t0 = time.time()
    tok_b.train_from_file(path, vocab_size, num_processes)
    print(f"train_from_file: vocab={vocab_size} in {time.time() - t0:.1f}s total")
    print(f"identical merges: {tok_a.merges == tok_b.merges}, "
          f"identical vocab: {tok_a.vocab == tok_b.vocab}")
    learned = [tok_b.vocab[i] for i in range(256, 256 + len(tok_b.merges))]
    print("longest learned token:", max(learned, key=len) if learned else None)


def train_bpe(input_path, vocab_size, special_tokens, num_processes=None):
    """Train a byte-level BPE tokenizer on a file (assignment 1, `train_bpe`).

    Thin wrapper over BPE.train_from_file that returns the shapes the
    assignment specifies:
      vocab:  dict[int, bytes]
      merges: list[tuple[bytes, bytes]], ordered by creation
    (train_from_file returns merges as integer token-id pairs.)
    """
    tokenizer = BPE(special_tokens=special_tokens)
    vocab, merge_ids = tokenizer.train_from_file(
        str(input_path), vocab_size, num_processes=num_processes)
    merges = [(vocab[left_id], vocab[right_id]) for left_id, right_id in merge_ids]
    return vocab, merges
