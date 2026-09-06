"""AdamW (Loshchilov & Hutter 2019), written on torch.optim.Optimizer.

Follows Algorithm 1 of the CS336 A1 handout: decoupled weight decay applied
before the moment update, bias-corrected step size, t starting at 1.
"""

from collections.abc import Callable
from typing import Optional

import torch


class AdamW ( torch.optim.Optimizer ) :
    def __init__ ( self, params , lr = 1e-3, betas = ( 0.9, 0.999) , eps = 1.e-8, weight_decay = 0.01 ) :
        if lr <0 : 
            raise ValueError ( f"Invalid learning rate: {lr}")
        if not 0 <= betas[0] < 1 or not 0 <= betas[1] < 1 :
            raise ValueError ( f"Invalid betas: {betas}" )
        if eps < 0 :
            raise ValueError ( f"Invalid epsilon: {eps}" )

        defaults = {"lr":lr, "betas":betas , "eps":eps , "weight_decay": weight_decay }
        

        super().__init__ ( params , defaults )


    def step ( self, closure: Optional[Callable ]= None ):
        loss = None if closure is None else closure ()

        for group in self.param_groups:
            lr = group["lr"]
            betas = group ["betas"]
            eps = group ["eps"]
            wdecay = group ["weight_decay"]

            for p in group ["params"] :

                if p.grad is None:
                    continue
                
                state = self.state [p]
                m = state.get("m", torch.zeros_like(p))
                v = state.get("v", torch.zeros_like(p))
                t  = state.get("t", 1 ) # get iteration number
                grad = p.grad.data
                lrt = lr *( 1 - betas[1]**t )**0.5 / ( 1- betas[0]**t)
                p.data  +=  -lr * wdecay * p.data
                m = betas[0] * m + ( 1- betas[0] ) * grad
                v = betas[1] * v + ( 1 - betas [1]) *grad**2
                p.data  +=  - lrt * m / ( v**0.5 + eps ) 

                state["t"] = t+1 
                state["m"] = m
                state["v"] = v

        return loss
