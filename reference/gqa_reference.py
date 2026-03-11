import torch
import torch.nn.functional as F
from einops import rearrange,einsum



def gqa_reference(Q, K, V, causal=False):
    """
    Q: [B, H_q, N, D]
    K: [B, H_kv, S, D]
    V: [B, H_kv, S, D]
    Returns: [B, H_q, S, D]
    """

    B, H_q, N, D = Q.shape
    B,H_kv,S,D=K.shape
    
    print("debug",B,H_q,N,D)
    #print("Debug", Q.shape, K.shape, V.shape)
    H_kv = K.shape[1]

    assert H_q % H_kv == 0, f"H_q ({H_q}) must be divisible by H_kv ({H_kv})"
    #find out number of groups
    num_groups = H_q // H_kv
    scale= D**0.5

    #reshape Q, K, V, always keep the elements you want to matrix multiply as the last two dimensions of the matrix as Batched Matrix Multiplication performs MM at the last two indices always.
    Q = Q.view(B, H_kv, num_groups, N, D) #b h_k g n d
    K = K.view(B, H_kv, S, D) #b h s d
    V = V.view(B, H_kv, S, D) #b h s d


    scores= einsum( Q, K,'b h_kv g n d, b h_kv s d -> b h_kv g n s')
    if causal:
        upper_triangular = torch.triu(torch.ones(S, S), diagonal=1)
        mask = upper_triangular.bool()
        scores = scores.masked_fill(mask, float('-inf'))

    attention= F.softmax(scores/scale, dim=-1)


    out= einsum( attention, V,'b h_kv g n s, b h_kv s d -> b h_kv g n d')

    print("Debug", out.shape)
    print("Debug", B, H_q, S, D)
    out= out.view(B, H_q, N, D)

    return out

    
    







  