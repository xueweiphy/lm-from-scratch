import torch


class Linear ( torch.nn.Module ) :

    def __init__ ( self, fan_in, fan_out, device = None, dtype = None ):
        super().__init__()
        self.w = torch.nn.Parameter(torch.empty(fan_in, fan_out, device=device, dtype=dtype)  )
        std = (  2./ ( fan_in + fan_out) )**0.5
        torch.nn.init.trunc_normal_(self.w, mean=0.0, std=std, a=-3.0 * std, b= 3.*std)


    def forward ( self, x  ) :
        out  = x @ self.w

        return out
        
#    def parameters(self):
#        return [self.w] 


class Embedding  ( torch.nn.Module )  :
    def __init__ ( self, vocab_size, embedding_dim, device = None, dtype = None ):
        super().__init__()
        self.Cmap  = torch.nn.Parameter ( torch.empty ( vocab_size, embedding_dim, device=device, dtype=dtype)  )
        torch.nn.init.trunc_normal_(self.Cmap, mean=0.0, std=1., a=-3.0 , b= 3.)

    def forward ( self, token_ids ):
        out = self.Cmap [ token_ids ]
        return out



class RmsNorm  ( torch.nn.Module ) :
    def __init__ ( self, d_model, eps : float = 1e-5,  device = None, dtype = None ):
        super().__init__()
        self.gamma = torch.nn.Parameter ( torch.ones (  d_model, device=device, dtype=dtype)  )
        self.d_model = d_model
        self.eps = eps

    def forward ( self, x  ):
        in_dtype = x.dtype
        x = x.to ( torch.float32) 
        ms = torch.mean(x * x, dim=-1, keepdim = True)
        rms = ( ms+ self.eps ) **0.5
        out = x / rms * self.gamma
        
        return out.to ( in_dtype ) 



class FFN_swiglu ( torch.nn.Module ) :
    def __init__ ( self, dmodel,  d_ff = None , device = None, dtype = None ) :
        super().__init__()
        if d_ff is None :
            d_ff = round ( 8 * dmodel / 3 / 64 ) * 64
        self.dmodel = dmodel
        self.dff    = d_ff
        self.W1 = Linear ( dmodel, d_ff, device=device, dtype=dtype )
        self.W3 = Linear ( dmodel, d_ff, device=device, dtype=dtype )
        self.W2 = Linear ( d_ff, dmodel, device=device, dtype=dtype )

        
    def forward ( self, x ) :
        y = self.W1 ( x) 
        out = self.W2 ( ( y* torch.sigmoid ( y ) *  self.W3 ( x)  )  )

        return out


class FFN_silu ( torch.nn.Module ) :
    def __init__ ( self, dmodel,  d_ff = None , device = None, dtype = None ) :
        super().__init__()
        if d_ff is None :
            d_ff =  4 * dmodel 
        self.dmodel = dmodel
        self.dff    = d_ff
        self.W1 = Linear ( dmodel, d_ff, device=device, dtype=dtype )
        self.W2 = Linear ( d_ff, dmodel, device=device, dtype=dtype )

        
    def forward ( self, x ) :
        y = self.W1 ( x) 
        out = self.W2 (  y* torch.sigmoid ( y )   )

        return out







class RoPE ( torch.nn.Module ) :
    def __init__ ( self, theta, d_k, max_seq_len , device = None )  :
        super().__init__ ()

        inv = theta** (- 2 * torch.arange ( d_k //2 ,  device = device) /d_k)
        thetalist  = torch.arange( max_seq_len ,  device = device).float().view ( -1, 1) @ inv.view ( 1, -1)
        #print ( thetalist.shape, inv.shape)
        #print ( thetalist )
        #self.coslist = thetalist.cos()
        #self.sinlist = thetalist.sin()
        self.register_buffer ( "coslist", thetalist.cos(), persistent = False ) 
        self.register_buffer ( "sinlist", thetalist.sin(), persistent = False ) 


    def forward ( self, x, token_positions ) :
        
        msin = self.sinlist [token_positions ]
        mcos = self.coslist [token_positions ]

        x = x.view ( *x.shape [:-1],  -1, 2 ) 
        x0, x1  = x[..., 0]  ,  x[..., 1]   

        o0 = x0 * mcos - x1 * msin
        o1 = x0 * msin + x1 * mcos

        return torch.stack ( [o0, o1], dim = -1 ).flatten ( -2 )



def softmax ( x, dim )  :
    xexp = (x- x.max ( dim =dim , keepdim = True).values ).exp()
    xexpsum = xexp.sum ( dim= dim , keepdim = True )
    out = xexp / xexpsum 
    return out


    
    
            

def Attention ( Qin , Kin , Vin , mask = None):
    Qshape = Qin.shape
    d_k = Qshape [-1]
    qk = Qin @ Kin.transpose ( -2, -1) * d_k **-0.5
    if mask is not None :        
        qk = qk.masked_fill(~mask, float("-inf"))

    out = softmax ( qk, dim = -1 ) @ Vin
    return out


class MultiheadSelfAttention ( torch.nn.Module) :
    def __init__ ( self,  d_model, num_heads  , device = None, dtype = None , max_seq_len = None, theta = None ) :
        super().__init__()
        d_k = d_model // num_heads
        d_v = d_model // num_heads
        self.num_heads = num_heads
        self.d_k = d_k
        self.d_v = d_v
        self.Wq =Linear (   d_model, num_heads * d_k , device=device, dtype=dtype )
        self.Wk =Linear (   d_model , num_heads * d_k , device=device, dtype=dtype )
        self.Wv =Linear (  d_model , num_heads * d_v , device=device, dtype=dtype )
        self.Wo = Linear ( num_heads * d_v, d_model  , device=device, dtype=dtype )

        self.rope = None
        if theta is not None :
            self.rope = RoPE  ( theta, d_k, max_seq_len , device = device ) 



    def forward ( self, xin, token_positions = None  )  :

        Qmulti = self.Wq ( xin )

        Kmulti = self.Wk ( xin )
        
        Vmulti = self.Wv ( xin  )
        

        seq_len = xin.shape[-2]
        mask = torch.ones ( seq_len, seq_len, dtype=torch.bool, device=xin.device ).tril()

        if token_positions is None :
            token_positions = torch.arange ( seq_len , device = xin.device ) 
        
        att = []
        for ii in range ( self.num_heads ) :
            
            Qi = Qmulti [ ..., ii * self.d_k : (ii +1) * self.d_k ]
            Ki = Kmulti [ ..., ii * self.d_k : (ii +1) * self.d_k ]
            Vi = Vmulti [ ..., ii * self.d_v : (ii +1) * self.d_v ]

            if self.rope is not None :
                Qi = self.rope ( Qi, token_positions ) 
                Ki = self.rope ( Ki, token_positions ) 
            
            att.append ( Attention ( Qi, Ki, Vi, mask ) )

            

        out = self.Wo ( torch.cat ( att, dim = -1 ))
        return out
        


class Transformer_block (torch.nn.Module) :
    def __init__ ( self, d_model, num_heads, d_ff ,eps : float = 1e-5, device = None,  dtype = None , max_seq_len = None, theta = None , norm = "rms" , norm_pos = "pre" , ffn = "swiglu" ):
        super().__init__()
        self.norm_pos = norm_pos
        self.norm1 = RmsNorm ( d_model, eps,  device , dtype ) if norm == "rms" else torch.nn.Identity()   # norm="none" -> layer_norm_ablation
        self.norm2 = RmsNorm ( d_model, eps,  device , dtype ) if norm == "rms" else torch.nn.Identity()
        self.mha = MultiheadSelfAttention  (   d_model, num_heads  , device = device, dtype = dtype , max_seq_len = max_seq_len, theta = theta)
        Ffn = FFN_swiglu if ffn == "swiglu" else FFN_silu                    # ffn="silu" -> swiglu_ablation
        self.ffn = Ffn ( d_model, d_ff = d_ff , device = device, dtype = dtype )



    def forward ( self, xin,  token_positions = None ) :

        if self.norm_pos == "pre" :                                    # x + f( norm(x) )
            x = xin + self.mha ( self.norm1 ( xin ) , token_positions = token_positions )
            y = x + self.ffn ( self.norm2 ( x ) )
        else :                                                         # norm( x + f(x) )  -- pre_norm_ablation
            x = self.norm1 ( xin + self.mha ( xin , token_positions = token_positions ) )
            y = self.norm2 ( x + self.ffn ( x ) )

        return y 




class Transformer_lm ( torch.nn.Module ) :
    def __init__ ( self, vocab_size,  context_length ,  num_layers, d_model, num_heads, d_ff ,
                   eps : float = 1e-5, device = None,  dtype = None , theta = None , norm = "rms" , norm_pos = "pre" , ffn = "swiglu" )  :

        super().__init__()
        self.cmap = Embedding ( vocab_size , d_model, device = device , dtype = dtype )
        
        self.block = torch.nn.ModuleList ( [ Transformer_block ( d_model, num_heads, d_ff ,eps, 
                                        device = device, dtype = dtype , max_seq_len = context_length, theta = theta , norm = norm , norm_pos = norm_pos , ffn = ffn )  
                                             for _ in range ( num_layers )  ] ) 

        self.norm = RmsNorm ( d_model, eps,  device , dtype ) if norm == "rms" else torch.nn.Identity()
        self.lin = Linear (  d_model, vocab_size, device , dtype )


    def forward ( self, xin  ) :
        positions = torch.arange ( xin.shape [-1 ], device = xin.device) 
        x = self.cmap ( xin )
        for bb in self.block :
            x = bb ( x, positions ) 

        x = self.norm (x)
        x = self.lin ( x ) 
        return x 




        




def cross_entropy ( logits, target ) :
    logits = logits.view ( -1,  logits.shape[-1] ) 
    target = target.view ( -1 )
    shifted = logits - logits.max ( dim = -1, keepdim = True ).values
    logsumexp = shifted.exp().sum ( dim = -1 ).log()
    logprob = shifted [ torch.arange ( target.shape[0] ), target ] - logsumexp
    return - logprob.mean()
