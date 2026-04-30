import torch

H = 7168
heads = 128
q_rope_dim = 64
q_nope_dim = 128
k_nope_dim = 128
k_rope_dim = 64
kv_heads = 1
kv_lora_dim = 512
q_lora_dim = 1536
v_dim = 128

seq_len = 16
cache_len = 128

device = "cuda" if torch.cuda.is_available() else "cpu"

dtype = torch.float32

def rmsnorm(x:torch.Tensor):
    return x/torch.sqrt(x.square().mean(dim=-1, keepdim=True)+1e-6)

def safe_softmax(P:torch.Tensor):
    M = torch.max(P,dim=-1,keepdim=True).values
    S = torch.exp(P - M)
    L = torch.sum(S, dim=-1,keepdim=True)
    return S/L, torch.log(L)+M

def get_rope_cos_sin(postions, d_rope, base=10000):
    freqs = 1/base**(torch.arange(0, d_rope, 2).float()/d_rope)
    freqs = torch.outer(postions, freqs)
    return freqs.cos(), freqs.sin()

def apply_rope(x, cos, sin):
    x1 = x[...,::2]
    x2 = x[...,1::2]
    o1 = x1*cos-x2*sin
    o2 = x1*sin+x2*cos
    return torch.stack([o1,o2],dim=-1).flatten(-2)

def mla_normal(W_QD, W_QU, W_KVD, W_KVU, W_V, X, KVC, KVR, postions):
    all_seq_len = seq_len + cache_len
    kv_postions = torch.arange(0, all_seq_len)
    q_postions = torch.arange(all_seq_len-seq_len,all_seq_len)
    freq_cos, freq_sin = get_rope_cos_sin(postions, q_rope_dim)
    q_lora = X @ W_QD
    q_lora = rmsnorm(q_lora)
    q_nope, q_pe = (q_lora @ W_QU).view(seq_len, heads, (q_nope_dim + q_rope_dim)).split([q_nope_dim,q_rope_dim], dim=-1)
    q_pe = apply_rope(q_pe, freq_cos.unsqueeze(-2), freq_sin.unsqueeze(-2))

    
    kv_lora, k_rope = (X @ W_KVD).split([kv_lora_dim,k_rope_dim],dim=-1)
    k_rope = apply_rope(k_rope, freq_cos, freq_sin)
    k_rope: torch.Tensor
    kv_lora = rmsnorm(kv_lora)
    kv_lora = torch.concat([KVC, kv_lora], dim=0)
    k_rope = torch.concat([KVR, k_rope], dim=0)
    k_nope, V = (kv_lora @ W_KVU).view(all_seq_len, heads, (k_nope_dim + v_dim)).split([k_nope_dim,v_dim], dim=-1)
    

    Q = torch.concat([q_nope, q_pe], dim=-1)
    K = torch.concat([k_nope,k_rope.unsqueeze(1).expand(-1,heads,-1)], dim=-1)

    scale = 1/(Q.shape[-1]**-0.5)
    # MHA
    P = torch.bmm(Q.transpose(0,1), K.permute(1,2,0)) * scale
    M = q_postions.unsqueeze(1) < kv_postions.unsqueeze(0)
    # print(M)
    P.masked_fill_(M.unsqueeze(0), float("-inf"))
    # print(P)
    S, LSE = safe_softmax(P)
    V_O = torch.bmm(S.to(dtype=dtype), V.transpose(0,1)).transpose(0,1)

    O = V_O.reshape(seq_len, -1) @ W_V

    return O, LSE.transpose(0,1)

def mla_absorbed(W_QD, W_QU, W_KVD, W_KVU, W_V, X, KVC, KVR, postions):
    W_KVU_Q,W_KVU_O =  W_KVU.view(kv_lora_dim, heads, k_nope_dim+v_dim).split([k_nope_dim, v_dim],dim=-1)
    all_seq_len = seq_len + cache_len
    kv_postions = torch.arange(0, all_seq_len)
    q_postions = torch.arange(all_seq_len-seq_len,all_seq_len)
    freq_cos, freq_sin = get_rope_cos_sin(postions, q_rope_dim)
    q_lora = X @ W_QD
    q_lora = rmsnorm(q_lora)
    q_nope, q_pe = (q_lora @ W_QU).view(seq_len, heads, (q_nope_dim + q_rope_dim)).split([q_nope_dim,q_rope_dim], dim=-1)
    q_pe = apply_rope(q_pe, freq_cos.unsqueeze(-2), freq_sin.unsqueeze(-2)).transpose(0,1)
    q_nope = torch.bmm(q_nope.transpose(0,1), W_KVU_Q.permute(1,2,0))

    
    kv_lora, k_rope = (X @ W_KVD).split([kv_lora_dim,k_rope_dim],dim=-1)
    k_rope = apply_rope(k_rope, freq_cos, freq_sin)
    k_rope: torch.Tensor
    kv_lora = rmsnorm(kv_lora)
    kv_lora = torch.concat([KVC, kv_lora], dim=0)
    k_rope = torch.concat([KVR, k_rope], dim=0)

    Q = torch.concat([q_nope, q_pe], dim=-1)
    K = torch.concat([kv_lora.unsqueeze(1).expand(-1,heads,-1),k_rope.unsqueeze(1).expand(-1,heads,-1)], dim=-1)

    scale = 1/(Q.shape[-1]**-0.5)
    # MHA
    P = torch.bmm(Q, K.permute(1,2,0)) * scale
    M = q_postions.unsqueeze(1) < kv_postions.unsqueeze(0)
    P.masked_fill_(M.unsqueeze(0), float("-inf"))
    S, LSE = safe_softmax(P)
    V_O = torch.bmm(S.to(dtype=dtype), K[..., :kv_lora_dim].transpose(0,1))
    V_O = torch.bmm(V_O, W_KVU_O.transpose(0,1)).transpose(0,1)

    O = V_O.reshape(seq_len, -1) @ W_V

    return O, LSE.transpose(0,1)



if __name__ == "__main__":
    W_QD = torch.randn(H, q_lora_dim, device=device, dtype=dtype)
    W_QU = torch.randn(q_lora_dim, heads*(q_rope_dim+q_nope_dim), device=device, dtype=dtype)

    W_KVD = torch.randn(H, kv_lora_dim + k_rope_dim, device=device, dtype=dtype)
    W_KVU = torch.randn(kv_lora_dim, heads*(k_nope_dim + v_dim), device=device, dtype=dtype)

    W_V = torch.randn(heads*v_dim, H, device=device, dtype=dtype)

    X = torch.randn(seq_len, H, device=device, dtype=dtype)
    KVC = torch.randn(cache_len, kv_lora_dim, device=device, dtype=dtype)
    KVR = torch.randn(cache_len, k_rope_dim, device=device, dtype=dtype)

    postions = torch.arange(cache_len, seq_len+cache_len)

    args = (W_QD, W_QU, W_KVD, W_KVU, W_V, X, KVC, KVR, postions)

    out1, lse1 = mla_normal(*args)
    out2, lse2 = mla_absorbed(*args)

    print(out1[:5,:8])
    print(out2[:5,:8])

    print(torch.allclose(out1,out2,1e-3,1e-3))
