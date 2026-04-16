import torch

def self_attention(Q,K,V,scale):
    Sij = Q@K.T*scale
    Pij = torch.softmax(Sij, dim=1)
    return Pij@V

SHM_SIZE = 1024*1024*16
def flash_attention_v2(Q,K,V,scale):
    Seqlen_q = Q.shape[0]
    Dim_q = Q.shape[1]

    Seqlen_kv = K.shape[0]
    Br = int(min(SHM_SIZE/4/Dim_q, Dim_q))
    Bc = int(SHM_SIZE/4/Dim_q)

    O = torch.zeros_like(Q, dtype=torch.float32,device="cuda:0")

    for i in range(0, Seqlen_q, Br):
        Qi = Q[i:i+Br,:]
        Mi = torch.zeros(Br,1,dtype=torch.float32,device="cuda:0")
        Mi = Mi.fill_(-float('inf'))
        Li = torch.zeros(Br,1,dtype=torch.float32,device="cuda:0")

        for j in range(0, Seqlen_kv, Bc):
            Kj = K[j:j+Bc,:]
            Vj = V[j:j+Bc,:]

            Sij = Qi@Kj.T*scale
            Mi_new = torch.maximum(Mi, torch.max(Sij,dim=1,keepdim=True)[0])
            Pij = torch.exp(Sij - Mi_new)
            Li = Li*torch.exp(Mi-Mi_new)+Pij.sum(1,keepdim=True)
            O[i:i+Br,:] = O[i:i+Br,:]*torch.exp(Mi-Mi_new) + Pij@Vj
            Mi = Mi_new
        O[i:i+Br,:] /= Li
    
    return O

if __name__ == "__main__":
    S = 32
    D = 32
    Q = torch.empty(S, D, dtype=torch.float32, device="cuda:0").uniform_(-1, 0)
    K = torch.empty(S, D, dtype=torch.float32, device="cuda:0").uniform_(-1, 0)
    V = torch.empty(S, D, dtype=torch.float32, device="cuda:0").uniform_(-1, 0)

    scale = D**-0.5

    O1 = self_attention(Q,K,V,scale)

    O2 = flash_attention_v2(Q,K,V,scale)

    print(f"{O1=} \n {O2=}")


    print(torch.allclose(O1,O2,1e-3,1e-3))