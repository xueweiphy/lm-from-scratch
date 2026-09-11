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

Vocab_size = int ( sys.argv[1] ) if len ( sys.argv ) > 1 else 10000
Text = os.path.join ( ROOT, "data", "arxiv_hepph_train.txt" )
Out = os.path.join ( ROOT, "experiments", f"arxiv_hepph_vocab{Vocab_size}.json" )

t0 = time.time()
tokenizer = BPE ( special_tokens = [ "<|endoftext|>" ] )
tokenizer.train_from_file ( Text, Vocab_size, verbose = True )
tokenizer.save ( Out )
print ( f"{Out}  ({time.time() - t0:.0f} s)" )

sample = "We compute the one-loop corrections to the Higgs boson mass in the MSSM."
for name, path in [ ( "tinystories", "tinystories_vocab10000.json" ), ( "arxiv", os.path.basename ( Out ) ) ] :
    ids = BPE.load ( os.path.join ( ROOT, "experiments", path ) ).encode ( sample )
    print ( f"{name:12s} {len(ids):3d} tokens for {len(sample)} chars" )
