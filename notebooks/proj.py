import torch
import itertools

N = 3
Nsq = N*N

basis = []

I_small = torch.eye(N)
I_big = torch.eye(Nsq)
J_small = torch.ones(N, N)
J_big = torch.ones(Nsq,Nsq)
L_small = J_small - I_small
L_big = J_big - I_big
IL = torch.kron(I_small,L_small)
J_gram = L_big-IL


base_row = [I_big, J_gram,IL]
base_column = [I_big, J_gram,IL]
base_number = [I_big, L_big]


for (P,Q,R) in itertools.product(base_row, base_column, base_number):
    B = torch.kron(P,torch.kron(Q,R))
    basis.append(B)
basis = torch.stack(basis)


def get_proj(C):
    ans = torch.zeros(729,729)
    for base in basis:
        base_inner = (base*base).sum()
        inter = base*C
        C_inner = (base*C).sum()
        ans += (C_inner/base_inner)*base
    return ans

def get_whole(S, proj_lr = 1):
    S.requires_grad = False
    S = S.cpu()
    C = S @ S.t()
    
    upper_mean = torch.mean(C[0, 1:730])
    new_row = torch.ones(1, 729)*upper_mean
    new_col = torch.ones(730, 1)*upper_mean
    new_col[0][0] = C[0][0]
    
    Sub = C[1:730, 1:730]
    ans = get_proj(Sub)
    # creating [[1, a], [a,ans]]
    new_C = torch.cat((new_row, ans), dim = 0)
    new_C = torch.cat((new_col, new_C), dim = 1)
    
    new_C = proj_lr*new_C + (1-proj_lr)*C
    
    U, D, Vh = torch.linalg.svd(new_C,full_matrices=True, out=None)
    new_S = U @ torch.sqrt(torch.diag(D))
    new_S.requires_grad = True
    return new_S