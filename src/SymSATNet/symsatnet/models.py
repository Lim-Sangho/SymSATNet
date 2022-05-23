import torch
import symsatnet._cuda
assert torch.cuda.is_available()


class EquivariantLayer(torch.autograd.Function):

    @staticmethod
    def forward(ctx, coeff: torch.Tensor, upper: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
        sub = torch.einsum("b,bij->ij", coeff, basis)
        upper_C = torch.cat([torch.zeros(1).cuda(), upper], dim = 0)
        sub_C = torch.cat([torch.unsqueeze(upper, 1), sub], dim = 1)
        C = torch.cat([torch.unsqueeze(upper_C, 0), sub_C], dim = 0)
        
        ctx.save_for_backward(basis)
        return C

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        basis, = ctx.saved_tensors
        grad_coeff = grad_upper = grad_bases = None

        grad_coeff = torch.einsum("bij,ij->b", basis, grad_output[1:basis.shape[1]+1, 1:basis.shape[2]+1])
        grad_upper = grad_output[0, 1:]

        return grad_coeff, grad_upper, grad_bases


class MixingFunc(torch.autograd.Function):

    @staticmethod
    def forward(ctx, C, z, is_input, max_iter, eps, prox_lam):
        B, n = z.size(0), C.size(0)
        k = 32 # int((2 * n) ** 0.5 + 3) // (4*4)
        ctx.prox_lam = prox_lam

        assert(C.is_cuda)
        device = 'cuda'

        ctx.g, ctx.gnrm = torch.zeros(B,k, device=device), torch.zeros(B,n, device=device)
        ctx.index = torch.zeros(B,n, dtype=torch.int, device=device)
        ctx.is_input = torch.zeros(B,n, dtype=torch.int, device=device)
        ctx.V, ctx.W = torch.zeros(B,n,k, device=device).normal_(), torch.zeros(B,k,n, device=device)
        ctx.z = torch.zeros(B,n, device=device)
        ctx.niter = torch.zeros(B, dtype=torch.int, device=device)

        ctx.C = torch.zeros(n,n, device=device)
        ctx.Cdiags = torch.zeros(n, device=device)

        ctx.z[:] = z.data
        ctx.C[:] = C.data
        ctx.is_input[:] = is_input.data

        perm = torch.randperm(n-1, dtype=torch.int, device=device)

        satnet_impl = symsatnet._cuda
        satnet_impl.init(perm, ctx.is_input, ctx.index, ctx.z, ctx.V)

        ctx.W[:] = ctx.V.transpose(1, 2)
        ctx.Cdiags[:] = torch.diagonal(C)

        satnet_impl.forward(max_iter, eps, 
                ctx.index, ctx.niter, ctx.C, ctx.z, 
                ctx.V, ctx.W, ctx.gnrm, ctx.Cdiags, ctx.g)

        return ctx.z.clone()
    
    @staticmethod
    def backward(ctx, dz):
        B, n = dz.size(0), ctx.C.size(0)
        k = 32 # int((2 * n) ** 0.5 + 3) // (4*4)

        assert(ctx.C.is_cuda)
        device = 'cuda'

        ctx.dC = torch.zeros(B,n,n, device=device)
        ctx.U, ctx.Phi = torch.zeros(B,n,k, device=device), torch.zeros(B,k,n, device=device)
        ctx.dz = torch.zeros(B,n, device=device)

        ctx.dz[:] = dz.data

        satnet_impl = symsatnet._cuda
        satnet_impl.backward(ctx.prox_lam, 
                ctx.is_input, ctx.index, ctx.niter, ctx.C, ctx.dC, ctx.z, ctx.dz,
                ctx.V, ctx.U, ctx.W, ctx.Phi, ctx.gnrm, ctx.Cdiags, ctx.g)

        ctx.dC = ctx.dC.sum(dim = 0)

        return ctx.dC, ctx.dz, None, None, None, None


class SymSATNet(torch.nn.Module):

    def __init__(self, n, bases, max_iter=40, eps=1e-4, prox_lam=1e-2):
        super(SymSATNet, self).__init__()
        assert n == bases.shape[1] == bases.shape[2]
        self.bases = bases.cuda()
        self.coeff = torch.nn.Parameter(torch.FloatTensor(len(self.bases)).normal_())
        self.upper = torch.nn.Parameter(torch.FloatTensor(n).normal_())
        self.max_iter, self.eps, self.prox_lam = max_iter, eps, prox_lam

    def forward(self, z, is_input):
        is_input = torch.cat([torch.ones(z.size(0), 1).cuda(), is_input], dim = 1)
        self.C = self.S = EquivariantLayer.apply(self.coeff, self.upper, self.bases)
        z = torch.cat([torch.ones(z.size(0), 1).cuda(), z], dim = 1)
        z = MixingFunc.apply(self.C, z, is_input, self.max_iter, self.eps, self.prox_lam)
        
        return z[:, 1:]


def main():
    is_cuda = True

    if is_cuda:
        gpu_num = 0
        torch.cuda.set_device(torch.device("cuda:{}".format(gpu_num)))
        print('Using', torch.cuda.get_device_name(gpu_num))
        print('Using', "cuda:{}".format(gpu_num))
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.cuda.init()

    n = 729
    aux = 0
    lr = 2e-2
    num_epoch = 100
    
    model = SymSATNet(n, n+1, aux).cuda()

    features = torch.load("sudoku/features.pt")
    labels = torch.load("sudoku/labels.pt")

    X = features[:100]
    Y = labels[:100]
    is_input = torch.sum(X, dim = -1)
    is_input = torch.tile(torch.unsqueeze(is_input, -1), (1, 1, 1, X.shape[-1]))

    X = X.reshape(X.shape[0], -1).cuda()
    Y = Y.reshape(Y.shape[0], -1).cuda()
    is_input = is_input.reshape(is_input.shape[0], -1).int().cuda()

    optimizer = torch.optim.Adam(model.parameters(), lr = lr)
    for epoch in range(num_epoch):
        optimizer.zero_grad()

        pred = model(X.contiguous(), is_input.contiguous())
        loss = torch.nn.BCELoss()(pred, Y)
        loss.backward()
        optimizer.step()
        
        if epoch % 10 == 0:
            print("Epoch: {} Loss: {}".format(epoch, loss.item()))

    # draw(model.S @ model.S.T)
    # draw(model.C)

if __name__ == "__main__":
    main()