"""Sample from a trained checkpoint (CS336 A1, decoding).

    python generate.py                                        # settings below
    python generate.py Prompt="Once upon a time" Temp=0.7 Topp=0.9 Nmax=400
"""
import sys
import torch

from model import Transformer_lm, softmax
from tokenizer import BPE

# ---- settings (override any of them as key=value on the command line) ----
Ckpt_path = "checkpoints/baseline.pt"
Vocab_path = "experiments/tinystories_vocab10000.json"
Vocab_size, Context_length, Dmodel, Num_layers, Num_heads, Theta = 10000, 256, 512, 4, 16, 10000
Prompt, Temp, Topp, Nmax, Device = "Long long time ago", 1.0, None, 400, "cpu"

for arg in sys.argv[1:] :
    key, value = arg.split ( "=", 1 )
    default = globals()[key]
    globals()[key] = value if isinstance ( default, str ) else type ( default or 1.0 ) ( value )
Dff = round ( 8 * Dmodel / 3 / 64 ) * 64
# ---------------------------------------------------------------------------


@torch.no_grad()
def generate ( model, ids, tokenizer, temp = 1.0, topp = 1.0, nmax = 400, context = 256 ) :
    """Sample continuations until <|endoftext|> or nmax new tokens.

    temp scales the logits before the softmax;  topp keeps the smallest set of
    tokens whose probabilities sum to topp (topp = 1.0 is plain sampling).
    """
    eos = tokenizer.encode ( "<|endoftext|>" ) [0]

    for _ in range ( nmax ) :
        logits = model ( ids [ -context : ], last_only = True )
        prob = softmax ( logits, dim = -1, temp = temp ).view ( -1 )

        psort, isort = torch.sort ( prob, descending = True )
        keep = int ( ( torch.cumsum ( psort, 0 ) - psort < topp ).sum() )      # crossing token kept
        next_id = isort [ torch.multinomial ( psort [ : keep ], num_samples = 1 ) ]

        ids = torch.cat ( [ ids, next_id ] )
        if next_id == eos :
            break

    return ids


if __name__ == "__main__" :
    tokenizer = BPE.load ( Vocab_path )
    model = Transformer_lm ( Vocab_size, Context_length, Num_layers, Dmodel, Num_heads, Dff,
                             device = Device, theta = Theta )
    model.load_state_dict ( torch.load ( Ckpt_path, map_location = Device ) ["model"] )

    ids = torch.tensor ( tokenizer.encode ( Prompt ), device = Device )
    out = generate ( model, ids, tokenizer, temp = Temp, topp = Topp or 1.0, nmax = Nmax,
                     context = Context_length )
    print ( tokenizer.decode ( out ) )
