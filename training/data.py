"""Random-window data loader over a flat token stream (CS336 A1, data_loading).

The corpus is one 1-D array of token IDs with <|endoftext|> IDs between
documents.  A training example is any window of `context_length` tokens;
its target is the same window shifted right by one.  Starts are drawn
uniformly (with replacement) from [0, len - context_length), so the target
window always fits.  `xin` may be a np.memmap — only the sampled windows are
copied, never the whole array.
"""

import torch


def data_loading ( xin, batch_size, context_length, device = None ):
    """
        xin : input numpy array with token IDs
        returns (inputs, targets), both torch.long of shape (batch_size, context_length)
    """
    out = torch.zeros ( ( batch_size, context_length), dtype = torch.long , device = device)
    target = torch.zeros ( ( batch_size, context_length), dtype = torch.long , device = device)

    lmax = len( xin )

    istart = torch.randint ( lmax - context_length , size = ( batch_size , ) )

    for ii in range ( batch_size ) :
        out[ii]  = torch.from_numpy ( xin [ istart[ii] : istart[ii] + context_length ].astype ( "int64" ) )
        target[ii]  = torch.from_numpy ( xin [ istart[ii] +1 : istart[ii] + context_length+1 ].astype ( "int64" ) )

    return out, target
