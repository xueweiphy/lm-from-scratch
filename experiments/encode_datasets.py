"""Encode a corpus into uint16 token IDs (assignment 1, 2.7(d)).

The whole file goes through BPE.encode as one string, so every <|endoftext|>
becomes its own ID in the stream — the document delimiter the data loader
(5.1) expects.  Reload with np.load(path, mmap_mode="r").
"""
import os, sys
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from tokenizer import BPE

DATA_DIR = os.path.join(ROOT, "data")
VOCAB = os.path.join(ROOT, "experiments", "tinystories_vocab10000.json")
VOCAB_ARXIV = os.path.join(ROOT, "experiments", "arxiv_hepph_vocab10000.json")


def encode_file(text_path, npy_path, vocab_path=VOCAB):
    tokenizer = BPE.load(vocab_path)
    with open(text_path) as file:
        text = file.read()
    ids = np.array(tokenizer.encode(text), dtype=np.uint16)
    np.save(npy_path, ids)
    print(f"{len(ids):,} tokens, max id {ids.max()}, {len(text.encode()) / len(ids):.2f} bytes/token")


CORPORA = [  # (text file, output .npy, vocab) — smallest first; TinyStories train takes ~2 h single-process
    ("TinyStoriesV2-GPT4-valid.txt", "tinystories_valid.npy", VOCAB),
    ("sample_50MB.txt", "tinystories_sample50MB.npy", VOCAB),
    ("TinyStoriesV2-GPT4-train.txt", "tinystories_train.npy", VOCAB),
    # arXiv: the TinyStories vocab for fine-tuning an existing checkpoint (the
    # embedding table is tied to it), the arXiv vocab for training from scratch.
    ("arxiv_hepph_valid.txt", "arxiv_hepph_valid.npy", VOCAB),
    ("arxiv_hepph_train.txt", "arxiv_hepph_train.npy", VOCAB),
    ("arxiv_hepph_valid.txt", "arxiv_hepph_nativevocab_valid.npy", VOCAB_ARXIV),
    ("arxiv_hepph_train.txt", "arxiv_hepph_nativevocab_train.npy", VOCAB_ARXIV),
]

if __name__ == "__main__":
    for text_name, npy_name, vocab in CORPORA:
        npy_path = f"{DATA_DIR}/{npy_name}"
        if os.path.exists(npy_path):
            print(f"{npy_name}: already exists, skipping")
            continue
        if not os.path.exists(f"{DATA_DIR}/{text_name}") or not os.path.exists(vocab):
            print(f"{npy_name}: missing input or vocab, skipping")
            continue
        print(f"{text_name} -> {npy_name}  [{os.path.basename(vocab)}]")
        encode_file(f"{DATA_DIR}/{text_name}", npy_path, vocab)
