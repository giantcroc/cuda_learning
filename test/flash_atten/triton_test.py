import triton
import triton.language as tl
import torch

@triton.jit
def flash_attention_v2_kernel(
    Q,K,V,O,
    scale,
    seq_q,seq_kv,
    stride_qo_head,stride_kv_head,
    stride_qo_seq,stride_kv_seq,
    stride_qo_head_dim,stride_kv_head_dim,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, HEAD_DIM: tl.constexpr,
):
    head_idx = tl.program_id(0)
    pid_m = tl.program_id(1)
    qo_seq_offset = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    head_dim_offset = tl.arange(0, HEAD_DIM)
    qo_offsets = head_idx*stride_qo_head + qo_seq_offset[:, None]*stride_qo_seq + head_dim_offset[None, :]*stride_qo_head_dim
    qo_mask = qo_seq_offset[:, None] < seq_q
    q = tl.load(Q + qo_offsets, mask=qo_mask, other=0.0)

    m = tl.zeros((BLOCK_M, 1), dtype=tl.float32) + float("-inf")
    l = tl.zeros((BLOCK_M,1), dtype=tl.float32) + 1
    o_accu = tl.zeros((BLOCK_M, HEAD_DIM), dtype=tl.float32)

    for kv_start in range(0, seq_kv, BLOCK_N):
        kv_seq_offset = kv_start + tl.arange(0, BLOCK_N)
        kv_mask = kv_seq_offset[:, None] < seq_kv
        kv_offsets = head_idx*stride_kv_head + kv_seq_offset[:, None]*stride_kv_seq + head_dim_offset[None, :]*stride_kv_head_dim
        k = tl.load(K + kv_offsets, mask=kv_mask, other=0.0)
        v= tl.load(V + kv_offsets, mask=kv_mask, other=0.0)

        k_t = tl.trans(k, (1,0))

        s = tl.dot(q, k_t) * scale

        s = tl.where(qo_seq_offset[:, None] + (seq_kv - seq_q) >= kv_seq_offset[None, :], s, float("-inf"))

        m_new = tl.maximum(m, tl.max(s, -1, keep_dims=True))

        p = tl.exp(s - m_new)

        l_new = l*tl.exp(m - m_new) + tl.sum(p, -1, keep_dims=True)

        o_accu = o_accu*tl.exp(m - m_new) + tl.dot(p.to(tl.float16), v)

        m = m_new
        l = l_new
    o_accu /= l
    tl.store(O + qo_offsets, o_accu, mask=qo_mask)

def flash_attention_v2(Q,K,V,BLOCK_M,BLOCK_N,scale):
    head_q, seq_q, head_dim = Q.shape
    seq_kv = K.shape[1]
    O = torch.empty_like(Q, dtype=Q.dtype, device=Q.device)
    grid = (head_q, triton.cdiv(seq_q, BLOCK_M), 1)
    flash_attention_v2_kernel[grid](
        Q,K,V,O,
        scale,
        seq_q,seq_kv,
        Q.stride(0),K.stride(0),
        Q.stride(1),K.stride(1),
        Q.stride(2),K.stride(2),
        BLOCK_M,BLOCK_N, head_dim,
    )
    return O

def flash_attention_v2_ref(Q,K,V,scale):
    seq_q = Q.shape[1]
    seq_kv = K.shape[1]
    S = torch.bmm(Q,K.permute(0,2,1)) * scale
    mask = torch.ones(seq_q,seq_kv, dtype=torch.bool, device=Q.device).tril(diagonal=seq_kv-seq_q)
    S.masked_fill_(mask.logical_not(), float("-inf"))
    P = torch.softmax(S, -1)
    return torch.bmm(P, V)

if __name__ == "__main__":
    HEADS = 4
    SEQ_Q = 32
    SEQ_KV = 64
    HEAD_DIM = 128
    BLOCK_M, BLOCK_N = 16, 16
    Q = torch.randn(HEADS, SEQ_Q, HEAD_DIM, dtype=torch.float16, device="cuda")
    K = torch.randn(HEADS, SEQ_KV, HEAD_DIM, dtype=torch.float16, device="cuda")
    V = torch.randn(HEADS, SEQ_KV, HEAD_DIM, dtype=torch.float16, device="cuda")

    scale = 1/HEAD_DIM**0.5

    O = flash_attention_v2(Q,K,V,BLOCK_M,BLOCK_N,scale)

    print(O)

    ref_O = flash_attention_v2_ref(Q,K,V,scale)
    print(ref_O)

    print(torch.allclose(O, ref_O, 1e-2, 1e-2))