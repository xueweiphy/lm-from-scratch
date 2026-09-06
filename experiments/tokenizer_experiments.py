"""Tokenizer experiments (assignment 1, section 2.7: tokenizer_experiments).

(a) Sample documents from TinyStories and OpenWebText, encode each with its
    own tokenizer, and report the compression ratio in bytes/token.
(b) Encode the same OpenWebText sample with the TinyStories tokenizer and
    compare.

A "document" is one <|endoftext|>-delimited chunk of a validation file.
Ratios are aggregate — total UTF-8 bytes divided by total tokens over the
whole sample, not the mean of the per-document ratios (those differ, and
the aggregate is what "bytes/token" conventionally means).
"""

import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
from tokenizer import BPE

DATA_DIR = os.path.join(ROOT, "data")
TINYSTORIES_FILE = f"{DATA_DIR}/TinyStoriesV2-GPT4-valid.txt"
OPENWEBTEXT_FILE = f"{DATA_DIR}/owt_valid.txt"
TINYSTORIES_VOCAB = os.path.join(HERE, "tinystories_vocab10000.json")
OPENWEBTEXT_VOCAB = os.path.join(HERE, "owt_vocab32000.json")

SPECIAL_TOKENS = ["<|endoftext|>"]
NUM_DOCUMENTS = 10
SEED = 42


def sample_documents(path, num_documents, seed):
    """Return `num_documents` documents drawn at random, without replacement."""
    with open(path, errors="ignore") as file:
        text = file.read()
    splitter = BPE(special_tokens=SPECIAL_TOKENS)
    documents = [chunk for chunk in splitter.split_on_special_tokens(text)
                 if chunk.strip()]
    return random.Random(seed).sample(documents, num_documents)


def compression_ratio(documents, tokenizer):
    """Aggregate counts over a list of documents.

    Returns (bytes, pretokens, tokens).  tokens/pretoken shows how often the
    tokenizer has to split a word: 1.0 means every pretoken is a single token
    (the vocabulary covers the corpus), and it climbs as the text drifts away
    from what the tokenizer was trained on.
    """
    num_bytes = sum(len(document.encode("utf-8")) for document in documents)
    num_pretokens = sum(len(tokenizer.pretokenize(document)) for document in documents)
    num_tokens = sum(len(tokenizer.encode(document)) for document in documents)
    return num_bytes, num_pretokens, num_tokens


def main():
    tinystories_tokenizer = BPE.load(TINYSTORIES_VOCAB)
    openwebtext_tokenizer = BPE.load(OPENWEBTEXT_VOCAB)

    tinystories_docs = sample_documents(TINYSTORIES_FILE, NUM_DOCUMENTS, SEED)
    openwebtext_docs = sample_documents(OPENWEBTEXT_FILE, NUM_DOCUMENTS, SEED)

    experiments = [
        ("TinyStories", "TinyStories 10K", tinystories_docs, tinystories_tokenizer),
        ("OpenWebText", "OpenWebText 32K", openwebtext_docs, openwebtext_tokenizer),
        ("OpenWebText", "TinyStories 10K", openwebtext_docs, tinystories_tokenizer),
        ("TinyStories", "OpenWebText 32K", tinystories_docs, openwebtext_tokenizer),
    ]

    print(f"{NUM_DOCUMENTS} documents per corpus, seed {SEED}\n")
    print(f"{'sample':<12} {'tokenizer':<17} {'bytes':>8} {'pretokens':>10} "
          f"{'tokens':>8} {'bytes/token':>12} {'tokens/pretoken':>16}")
    print("-" * 89)
    for sample_name, tokenizer_name, documents, tokenizer in experiments:
        num_bytes, num_pretokens, num_tokens = compression_ratio(documents, tokenizer)
        print(f"{sample_name:<12} {tokenizer_name:<17} {num_bytes:>8} "
              f"{num_pretokens:>10} {num_tokens:>8} "
              f"{num_bytes / num_tokens:>12.3f} "
              f"{num_tokens / num_pretokens:>16.3f}")


if __name__ == "__main__":
    main()
