"""Train a byte-level BPE on the arXiv abstracts (physics vocabulary, not children's stories).

    python experiments/train_arxiv_bpe.py                       # 10000 merges on the hep-ph train file
    python experiments/train_arxiv_bpe.py 16000                 # a bigger vocabulary

CPU only, uses all cores; ~10 min for 10M characters.  Writes
experiments/arxiv_hepph_vocab<size>.json in the same format BPE.load reads.
"""
import os, sys, time

ROOT = os.path.dirname ( os.path.dirname ( os.path.abspath ( __file__ ) ) )
sys.path.insert ( 0, ROOT )
from tokenizer import BPE

Sample = "We compute the one-loop corrections to the Higgs boson mass in the MSSM."


def main () :                         # multiprocessing spawns children that re-import this file,
    vocab_size = int ( sys.argv[1] ) if len ( sys.argv ) > 1 else 10000   # so nothing may run at import time
    text = os.path.join ( ROOT, "data", "arxiv_hepph_train.txt" )
    out = os.path.join ( ROOT, "experiments", f"arxiv_hepph_vocab{vocab_size}.json" )

    t0 = time.time()
    tokenizer = BPE ( special_tokens = [ "<|endoftext|>" ] )
    tokenizer.train_from_file ( text, vocab_size, verbose = True )
    tokenizer.save ( out )
    print ( f"{out}  ({time.time() - t0:.0f} s)" )

    for name, path in [ ( "tinystories", "tinystories_vocab10000.json" ), ( "arxiv", os.path.basename ( out ) ) ] :
        ids = BPE.load ( os.path.join ( ROOT, "experiments", path ) ).encode ( Sample )
        print ( f"{name:12s} {len(ids):3d} tokens for {len(Sample)} chars" )


if __name__ == "__main__" :
    main()
