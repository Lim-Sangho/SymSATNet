from __future__ import annotations
from abc import ABCMeta, abstractmethod
from itertools import product
from dataset.cube_generator import Cube333
from utils.algebra import *
import torch


class Grammar(metaclass = ABCMeta):
    """
    Grammar of permutation groups.
    """

    @abstractmethod
    def __init__(self, *args) -> None:
        pass

    @abstractmethod
    def __str__(self) -> str:
        pass

    @abstractmethod
    def _proj(self, C: torch.Tensor) -> torch.Tensor:
        """
        Return the projected matrix of C.
        """
        pass

    @abstractmethod
    def basis(self) -> torch.Tensor:
        """
        Return the basis matrices.
        """
        pass

    @abstractmethod
    def orbits(self) -> torch.Tensor:
        """
        Return the tensor with row vectors of orbits.
        """
        pass

    @abstractmethod
    def n_basis(self) -> int:
        """
        Return the size of basis.
        """
        pass

    @abstractmethod
    def n_orbits(self) -> int:
        """
        Return the size of orbits.
        """
        pass

    def symmetrize(self) -> Grammar:
        """
        Make the basis symmetric.
        Return itself.
        """
        basis = torch.stack([(basis + basis.T).flatten() for basis in self.basis()])
        basis = gram_schmidt(gram_schmidt(basis))
        basis = basis.reshape(-1, self.dim, self.dim)
        self.basis_cached = basis

        return self


class Id(Grammar):
    """
    Class of identity groups.
    """
    
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim
        self.basis_cached = None
        self.orbits_cached = None

    def __str__(self) -> str:
        return f"I{self.dim}"

    def _proj(self, C: torch.Tensor) -> torch.Tensor:
        return C

    def _forward(self, coeff: torch.Tensor) -> torch.Tensor:
        C = coeff.reshape(self.dim, self.dim, *coeff.shape[1:])
        return C

    def _backward(self, grad_C: torch.Tensor, option: Optional[str] = 'grad') -> torch.Tensor:
        grad_coeff = grad_C.flatten(0, 1)
        return grad_coeff

    def _coeff_diag_index(self) -> torch.Tensor:
        return torch.arange(self.dim) * (self.dim + 1)

    def cuda(self) -> Grammar:
        self.orbits_cached = self.orbits().cuda()
        return self

    def basis(self) -> torch.Tensor:
        if self.basis_cached is None:
            self.basis_cached = torch.eye(self.dim ** 2).reshape(-1, self.dim, self.dim)
        return self.basis_cached

    def orbits(self) -> torch.Tensor:
        if self.orbits_cached is None:
            self.orbits_cached = torch.eye(self.dim)
        return self.orbits_cached

    def n_basis(self) -> int:
        return self.dim ** 2

    def n_orbits(self) -> int:
        return self.dim


class Cyclic(Grammar):
    """
    Class of cyclic groups.
    """

    def __init__(self, dim: int) -> None:
        self.dim = dim
        self.basis_cached = None
        self.orbits_cached = None

    def __str__(self) -> str:
        return f"C{self.dim}"

    def _proj(self, C: torch.Tensor) -> torch.Tensor:
        vector_inner = torch.einsum("ijk,jk...->i...", self.basis(), C)
        basis_inner = torch.einsum("i...->i", self.basis())
        coeffs = torch.einsum("i...,i->i...", vector_inner, 1 / basis_inner)
        C_proj = torch.einsum("ijk,i...->jk...", self.basis(), coeffs)

        return C_proj

    def _forward(self, coeff: torch.Tensor) -> torch.Tensor:
        C = torch.einsum("i...,ijk->jk...", coeff, self.basis())
        return C

    def _backward(self, grad_C: torch.Tensor, option: Optional[str] = 'grad') -> torch.Tensor:
        if option == 'grad':
            grad_coeff = torch.einsum("ijk,jk...->i...", self.basis(), grad_C)
        elif option == 'coeff':
            grad_coeff = coordinates(grad_C, self.basis())
        return grad_coeff

    def _coeff_diag_index(self) -> torch.Tensor:
        return torch.LongTensor([0])

    def cuda(self) -> Grammar:
        self.basis_cached = self.basis().cuda()
        self.orbits_cached = self.orbits().cuda()
        return self

    def basis(self) -> torch.Tensor:
        if self.basis_cached is None:
            self.basis_cached = torch.stack([torch.roll(torch.eye(self.dim), i, 0) for i in range(self.dim)])
        return self.basis_cached

    def orbits(self) -> torch.Tensor:
        if self.orbits_cached is None:
            self.orbits_cached = torch.ones(1, self.dim)
        return self.orbits_cached

    def n_basis(self) -> int:
        return self.dim

    def n_orbits(self) -> int:
        return 1
        

class Symm(Grammar):
    """
    Class of symmetric groups.
    """

    def __init__(self, dim: int) -> None:
        self.dim = dim
        self.basis_cached = None
        self.orbits_cached = None

    def __str__(self) -> str:
        return f"S{self.dim}"

    def _proj(self, C: torch.Tensor) -> torch.Tensor:
        vector_inner = torch.einsum("ijk,jk...->i...", self.basis(), C)
        basis_inner = torch.einsum("i...->i", self.basis())
        coeffs = torch.einsum("i...,i->i...", vector_inner, 1 / basis_inner)
        C_proj = torch.einsum("ijk,i...->jk...", self.basis(), coeffs)

        return C_proj

    def _forward(self, coeff: torch.Tensor) -> torch.Tensor:
        C = torch.einsum("i...,ijk->jk...", coeff, self.basis())
        return C

    def _backward(self, grad_C: torch.Tensor, option: Optional[str] = 'grad') -> torch.Tensor:
        if option == 'grad':
            grad_coeff = torch.einsum("ijk,jk...->i...", self.basis(), grad_C)
        elif option == 'coeff':
            grad_coeff = coordinates(grad_C, self.basis())
        return grad_coeff

    def _coeff_diag_index(self) -> torch.Tensor:
        return torch.LongTensor([0])

    def cuda(self) -> Grammar:
        self.basis_cached = self.basis().cuda()
        self.orbits_cached = self.orbits().cuda()
        return self

    def basis(self) -> torch.Tensor:
        if self.basis_cached is None:
            basis_1 = torch.eye(self.dim)
            basis_2 = torch.ones(self.dim, self.dim) - torch.eye(self.dim)
            self.basis_cached = torch.stack([basis_1, basis_2])

        return self.basis_cached
    
    def orbits(self) -> torch.Tensor:
        if self.orbits_cached is None:
            self.orbits_cached = torch.ones(1, self.dim)
        return self.orbits_cached

    def n_basis(self) -> int:
        return 2

    def n_orbits(self) -> int:
        return 1


class Gen(Grammar):
    """
    Class of groups generated by some given generators.
    """

    def __init__(self, dim: int, generators: torch.Tensor) -> None:
        assert generators.shape[1] == dim
        self.dim = dim
        self.generators = generators.long()
        self.basis_cached = None
        self.orbits_cached = None

    def __str__(self) -> str:
        return f"<{self.generators.tolist()}>"

    def _proj(self, C: torch.Tensor) -> torch.Tensor:
        vector_inner = torch.einsum("ijk,jk...->i...", self.basis(), C)
        basis_inner = torch.einsum("i...->i", self.basis())
        coeffs = torch.einsum("i...,i->i...", vector_inner, 1 / basis_inner)
        C_proj = torch.einsum("ijk,i...->jk...", self.basis(), coeffs)

        return C_proj

    def _forward(self, coeff: torch.Tensor) -> torch.Tensor:
        C = torch.einsum("i...,ijk->jk...", coeff, self.basis())
        return C

    def _backward(self, grad_C: torch.Tensor, option: Optional[str] = 'grad') -> torch.Tensor:
        if option == 'grad':
            grad_coeff = torch.einsum("ijk,jk...->i...", self.basis(), grad_C)
        elif option == 'coeff':
            grad_coeff = coordinates(grad_C, self.basis())
        return grad_coeff

    def _coeff_diag_index(self) -> torch.Tensor:
        return torch.LongTensor([i for i, b in enumerate(self.basis()) 
            if torch.allclose(b * (torch.ones(b.shape) - torch.eye(b.shape[0])), torch.zeros(1))])

    def cuda(self) -> Grammar:
        self.basis_cached = self.basis().cuda()
        self.orbits_cached = self.orbits().cuda()
        return self

    def basis(self) -> torch.Tensor:
        if self.basis_cached is None:
            I = torch.eye(self.dim)
            rho = torch.cat([torch.kron(I[perm], I[perm]) - torch.kron(I, I) for perm in self.generators])
            U, S, V = torch.svd(rho)
            is_null = torch.isclose(S, torch.zeros(1), atol = torch.norm(rho) * 1e-6)
            self.basis_cached = V.T[is_null].reshape(-1, self.dim, self.dim)

        return self.basis_cached
    
    def orbits(self) -> torch.Tensor:
        if self.orbits_cached is None:
            remain = list(range(self.dim))
            orbits = []
            while len(remain) > 0:
                prev = set()
                orbit = {remain[0]}
                while prev != orbit:
                    prev = orbit.copy()
                    for e, perm in product(orbit, self.generators):
                        orbit.add(perm[e].item())
                remain = list(set(remain) - orbit)
                one_hot = torch.zeros(self.dim)
                one_hot[list(orbit)] = 1
                orbits.append(one_hot)
            self.orbits_cached = torch.stack(orbits)

        return self.orbits_cached

    def n_basis(self) -> int:
        return len(self.basis())

    def n_orbits(self) -> int:
        return len(self.orbits())


class Sum(Grammar):
    """
    Class of direct sums of groups.
    """

    def __init__(self, G1: Grammar, G2: Grammar) -> None:
        self.G1 = G1
        self.G2 = G2
        self.dim = G1.dim + G2.dim
        self.basis_cached = None
        self.orbits_cached = None

    def __str__(self) -> str:
        return f"({self.G1} \u2295 {self.G2})"

    def _proj(self, C: torch.Tensor) -> torch.Tensor:
        dim_1 = self.G1.dim
        C_proj = torch.zeros(C.shape)
        C_proj[:dim_1, :dim_1] = self.G1._proj(C[:dim_1, :dim_1])
        C_proj[dim_1:, dim_1:] = self.G2._proj(C[dim_1:, dim_1:])
        C_proj[:dim_1, dim_1:] = proj_orbit(C[:dim_1, dim_1:], self.G1, self.G2)
        C_proj[dim_1:, :dim_1] = proj_orbit(C[dim_1:, :dim_1], self.G2, self.G1)

        return C_proj

    def _forward(self, coeff: torch.Tensor) -> torch.Tensor:
        dim_1 = self.G1.dim
        b1 = self.G1.n_basis()
        b2 = self.G2.n_basis()
        o1 = self.G1.n_orbits()
        o2 = self.G2.n_orbits()

        C = torch.zeros(self.dim, self.dim, *coeff.shape[1:])
        if coeff.is_cuda: C = C.cuda()
        
        C[:dim_1, :dim_1] = self.G1._forward(coeff[:b1])
        C[dim_1:, dim_1:] = self.G2._forward(coeff[b1:b1+b2])
        C[:dim_1, dim_1:] = forward_orbit(coeff[b1+b2:b1+b2+(o1*o2)], self.G1, self.G2)
        C[dim_1:, :dim_1] = forward_orbit(coeff[b1+b2+(o1*o2):], self.G2, self.G1)

        return C

    def _backward(self, grad_C: torch.Tensor, option: Optional[str] = 'grad') -> torch.Tensor:
        dim_1 = self.G1.dim
        grad_1 = self.G1._backward(grad_C[:dim_1, :dim_1], option)
        grad_2 = self.G2._backward(grad_C[dim_1:, dim_1:], option)
        grad_3 = backward_orbit(grad_C[:dim_1, dim_1:], self.G1, self.G2, option)
        grad_4 = backward_orbit(grad_C[dim_1:, :dim_1], self.G2, self.G1, option)

        return torch.cat([grad_1, grad_2, grad_3, grad_4])

    def _coeff_diag_index(self) -> torch.Tensor:
        return torch.cat([self.G1._coeff_diag_index(), self.G2._coeff_diag_index() + self.G1.n_basis()])

    def cuda(self) -> Grammar:
        self.G1 = self.G1.cuda()
        self.G2 = self.G2.cuda()
        self.orbits_cached = self.orbits().cuda()
        return self

    def basis(self) -> torch.Tensor:
        if self.basis_cached is None:
            dim_1 = self.G1.dim
            dim_2 = self.G2.dim
            basis = []
            for B in self.G1.basis():
                basis.append(torch.block_diag(B, torch.zeros(dim_2, dim_2)))
            for B in self.G2.basis():
                basis.append(torch.block_diag(torch.zeros(dim_1, dim_1), B))
            for orbit_1, orbit_2 in product(self.G1.orbits(), self.G2.orbits()):
                total = torch.zeros(dim_1 + dim_2, dim_1 + dim_2)
                row = torch.zeros(dim_1, dim_2)
                column = torch.zeros(dim_1, dim_2)
                row[orbit_1.bool()] = 1
                column[:,orbit_2.bool()] = 1
                total[:dim_1, dim_1:] = row * column
                basis.append(total)
                basis.append(total.T)
            self.basis_cached = torch.stack(basis)

        return self.basis_cached

    def orbits(self) -> torch.Tensor:
        if self.orbits_cached is None:
            dim_1 = self.G1.dim
            dim_2 = self.G2.dim
            orbits = []
            for orbit in self.G1.orbits():
                total = torch.zeros(dim_1 + dim_2)
                total[:dim_1] = orbit
                orbits.append(total)
            for orbit in self.G2.orbits():
                total = torch.zeros(dim_1 + dim_2)
                total[dim_1:] = orbit
                orbits.append(total)
            self.orbits_cached = torch.stack(orbits)

        return self.orbits_cached

    def n_basis(self) -> int:
        return self.G1.n_basis() + self.G2.n_basis() + (self.G1.n_orbits() * self.G2.n_orbits() * 2)

    def n_orbits(self) -> int:
        return self.G1.n_orbits() + self.G2.n_orbits()


class Kron(Grammar):
    """
    Class of Kronecker products of groups.
    """

    def __init__(self, G1: Grammar, G2: Grammar) -> None:
        self.G1 = G1
        self.G2 = G2
        self.dim = G1.dim * G2.dim
        self.basis_cached = None
        self.orbits_cached = None

    def __str__(self) -> str:
        return f"({self.G1} \u2297 {self.G2})"

    def _proj(self, C: torch.Tensor) -> torch.Tensor:
        dim_1 = self.G1.dim
        dim_2 = self.G2.dim
        C_hat = torch.stack([torch.stack([C[i*dim_2:(i+1)*dim_2, j*dim_2:(j+1)*dim_2] for j in range(dim_1)]) for i in range(dim_1)])
        for i, j in product(range(dim_1), range(dim_1)):
            C_hat[i, j] = self.G2._proj(C_hat[i, j])
        C_hat = self.G1._proj(C_hat)
        
        C_proj = torch.zeros(C.shape)
        for i, j in product(range(dim_1), range(dim_1)):
            C_proj[i*dim_2:(i+1)*dim_2, j*dim_2:(j+1)*dim_2] = C_hat[i,j]
            
        return C_proj

    def _forward(self, coeff: torch.Tensor) -> torch.Tensor:
        coeff_2 = coeff.reshape(self.G1.n_basis(), self.G2.n_basis(), *coeff.shape[1:]).transpose(0, 1)
        coeff_1 = self.G2._forward(coeff_2).transpose(1, 2).transpose(0, 1)
        C = self.G1._forward(coeff_1).transpose(1, 2)
        return C.reshape(self.dim, self.dim, *coeff.shape[1:])

    def _backward(self, grad_C: torch.Tensor, option: Optional[str] = 'grad') -> torch.Tensor:
        dim_1 = self.G1.dim
        dim_2 = self.G2.dim
        C_hat = torch.stack([torch.stack([grad_C[i*dim_2:(i+1)*dim_2, j*dim_2:(j+1)*dim_2] for j in range(dim_1)]) for i in range(dim_1)])
        C_proj_1 = self.G1._backward(C_hat, option).transpose(0, 1).transpose(1, 2)
        C_proj_2 = self.G2._backward(C_proj_1, option).transpose(0, 1)

        return C_proj_2.flatten(0, 1)

    def _coeff_diag_index(self) -> torch.Tensor:
        c1 = self.G1._coeff_diag_index()
        c2 = self.G2._coeff_diag_index()
        b1 = self.G1.n_basis()
        b2 = self.G2.n_basis()
        return torch.arange(b1 * b2).reshape(b1, b2)[c1][:,c2].flatten()

    def cuda(self) -> Grammar:
        self.G1 = self.G1.cuda()
        self.G2 = self.G2.cuda()
        self.orbits_cached = self.orbits().cuda()
        return self

    def basis(self) -> torch.Tensor:
        if self.basis_cached is None:
            self.basis_cached = torch.stack([torch.kron(B1, B2)
                for B1, B2 in product(self.G1.basis(), self.G2.basis())])
        return self.basis_cached

    def orbits(self) -> torch.Tensor:
        if self.orbits_cached is None:
            self.orbits_cached = torch.stack([torch.kron(orbit_1, orbit_2)
                for orbit_1, orbit_2 in product(self.G1.orbits(), self.G2.orbits())])
        return self.orbits_cached

    def n_basis(self) -> int:
        return self.G1.n_basis() * self.G2.n_basis()
    
    def n_orbits(self) -> int:
        return self.G1.n_orbits() * self.G2.n_orbits()


class Wreath(Grammar):
    """
    Class of wreath products of groups.
    """

    def __init__(self, G1: Grammar, G2: Grammar) -> None:
        self.G1 = G1
        self.G2 = G2
        self.dim = G1.dim * G2.dim
        self.basis_cached = None
        self.orbits_cached = None

    def __str__(self) -> str:
        return f"({self.G1} \u2240 {self.G2})"

    def _proj(self, C: torch.Tensor) -> torch.Tensor:
        dim_1 = self.G1.dim
        dim_2 = self.G2.dim
        C_hat = torch.stack([torch.stack([C[i*dim_1:(i+1)*dim_1, j*dim_1:(j+1)*dim_1] for j in range(dim_2)]) for i in range(dim_2)])
        for i, j in product(range(dim_2), range(dim_2)):
            if i == j:
                C_hat[i, j] = self.G1._proj(C_hat[i, j])
            else:
                C_hat[i, j] = proj_orbit(C_hat[i, j], self.G1, self.G1)
        C_hat = self.G2._proj(C_hat)
        
        C_proj = torch.zeros(C.shape)
        for i, j in product(range(dim_2), range(dim_2)):
            C_proj[i*dim_1:(i+1)*dim_1, j*dim_1:(j+1)*dim_1] = C_hat[i,j]
            
        return C_proj

    def _forward(self, coeff: torch.Tensor) -> torch.Tensor:
        b1 = self.G1.n_basis()
        b2 = self.G2.n_basis()
        o1 = self.G1.n_orbits()
        o2 = self.G2.n_orbits()
        c2 = self.G2._coeff_diag_index()

        diag = coeff[:b1*o2]
        diag = forward_orbit(diag, self.G2, Id(dim = b1)).transpose(0, 1)
        diag = self.G1._forward(diag)
        diag_embed = torch.zeros(diag.shape[0], diag.shape[1], diag.shape[2], diag.shape[2], *diag.shape[3:])
        if coeff.is_cuda: diag_embed = diag_embed.cuda()
        for i in range(diag.shape[2]):
            diag_embed[:, :, i, i, ...] = diag[:, :, i, ...]

        off_diag = coeff[b1*o2:].reshape(b2-o2, o1**2, *coeff.shape[1:])
        for i in c2:
            pad = torch.zeros(1, *off_diag.shape[1:])
            if coeff.is_cuda: pad = pad.cuda()
            off_diag = torch.cat([off_diag[:i], pad, off_diag[i:]])
        off_diag = self.G2._forward(off_diag).transpose(1, 2).transpose(0, 1)
        off_diag = forward_orbit(off_diag, self.G1, self.G1)
        
        C_hat = (diag_embed + off_diag).transpose(0, 2).transpose(1, 3).transpose(1, 2)
        return C_hat.reshape(self.dim, self.dim, *C_hat.shape[4:])

    def _backward(self, grad_C: torch.Tensor, option: Optional[str] = 'grad') -> torch.Tensor:
        dim_1 = self.G1.dim
        dim_2 = self.G2.dim
        b1 = self.G1.n_basis()
        c2 = self.G2._coeff_diag_index()
        C_hat = torch.stack([torch.stack([grad_C[i*dim_1:(i+1)*dim_1, j*dim_1:(j+1)*dim_1] for j in range(dim_2)]) for i in range(dim_2)]).transpose(0, 2).transpose(1, 3)

        diag = self.G1._backward(torch.einsum("ijkk...->ijk...", C_hat), option).transpose(0, 1)
        diag = backward_orbit(diag, self.G2, Id(dim = b1), option)

        off_diag = C_hat - torch.diag_embed(torch.diagonal(C_hat, dim1 = 2, dim2 = 3), dim1 = 2, dim2 = 3)
        off_diag = backward_orbit(off_diag, self.G1, self.G1, option).transpose(0, 1).transpose(1, 2)
        off_diag = self.G2._backward(off_diag, option)
        off_diag = torch.stack([off_diag[i] for i in range(len(off_diag)) if i not in c2]).flatten(0, 1)

        return torch.cat([diag, off_diag])

    def _coeff_diag_index(self) -> torch.Tensor:
        c1 = self.G1._coeff_diag_index()
        b1 = self.G1.n_basis()
        o2 = self.G2.n_orbits()
        return torch.arange(o2 * b1).reshape(o2, b1)[:,c1].flatten()
        
    def cuda(self) -> Grammar:
        self.G1 = self.G1.cuda()
        self.G2 = self.G2.cuda()
        self.orbits_cached = self.orbits().cuda()
        return self

    def basis(self) -> torch.Tensor:
        if self.basis_cached is None:
            dim_1 = self.G1.dim
            dim_2 = self.G2.dim
            basis = []
            for B in self.G1.basis():
                basis.append(torch.kron(torch.eye(dim_2), B)) # TODO: Fix it correctly
            for B in self.G2.basis():
                if not torch.allclose(B * (torch.ones(B.shape) - torch.eye(B.shape[0])), torch.zeros(1)):
                    basis.append(torch.kron(B, torch.ones(dim_1, dim_1)))
            self.basis_cached = torch.stack(basis)

        return self.basis_cached

    def orbits(self) -> torch.Tensor:
        if self.orbits_cached is None:
            self.orbits_cached = torch.stack([torch.kron(orbit_1, orbit_2)
                for orbit_1, orbit_2 in product(self.G1.orbits(), self.G2.orbits())])
        return self.orbits_cached

    def n_basis(self) -> int:
        return self.G1.n_basis() * self.G2.n_orbits() + (self.G2.n_basis() - self.G2.n_orbits()) * (self.G1.n_orbits() ** 2)

    def n_orbits(self) -> int:
        return self.G1.n_orbits() * self.G2.n_orbits()


class Sudoku(Kron):
    """
    Class of the Sudoku group.
    """

    def __init__(self) -> None:
        perm_row_column = Wreath(Symm(dim = 3), Symm(dim = 3))
        perm_number = Symm(dim = 9)
        super().__init__(Kron(perm_row_column, perm_row_column), perm_number)


class Cube(Kron):
    """
    Class of the Rubik's cube group.
    """
    
    def __init__(self) -> None:
        cube = Cube333()
        moves = [["U"], ["D"], ["F"], ["B"], ["L"], ["R"], ["M"], ["E"], ["S"]]
        perm_move = []
        for move in moves:
            cube.move(move)
            perm_move.append(torch.Tensor(cube.index_state.flatten()))
            cube.reset()
        perm_move = torch.stack(perm_move)
        perm_move = Gen(dim = 54, generators = perm_move)

        perm_color = torch.Tensor([[4, 0, 2, 3, 5, 1], [2, 1, 5, 0, 4, 3], [0, 2, 4, 1, 3, 5]])
        perm_color = Gen(dim = 6, generators = perm_color)
        super().__init__(perm_move, perm_color)


def proj_orbit(C: torch.Tensor, G1: Grammar, G2: Grammar) -> torch.Tensor:
    """
    Project the off-diagonal matrix C with the actions of G1 and G2.
    The shape of C is [G1.dim, G2.dim, ... ].
    """
    C_proj = C

    if not isinstance(G1, Id):
        orbit_1 = G1.orbits()
        C_inner_1 = torch.einsum("ij,j...->i...", orbit_1, C)
        orbit_inner_1 = torch.einsum("i...->i", orbit_1)
        coeffs_1 = torch.einsum("i...,i->i...", C_inner_1, 1 / orbit_inner_1)
        C_proj = torch.einsum("ij,i...->j...", orbit_1, coeffs_1)
    C_proj = C_proj.transpose(0, 1)

    if not isinstance(G2, Id):
        orbit_2 = G2.orbits()
        C_inner_2 = torch.einsum("ij,j...->i...", orbit_2, C_proj)
        orbit_inner_2 = torch.einsum("i...->i", orbit_2)
        coeffs_2 = torch.einsum("i...,i->i...", C_inner_2, 1 / orbit_inner_2)
        C_proj = torch.einsum("ij,i...->j...", orbit_2, coeffs_2)
    C_proj = C_proj.transpose(0, 1)

    return C_proj


def forward_orbit(coeff: torch.Tensor, G1: Grammar, G2: Grammar) -> torch.Tensor:
    """
    Compute the off-diagonals in $G1 \oplus G2$ with the coefficient.
    """
    C = coeff.reshape(G1.n_orbits(), G2.n_orbits(), *coeff.shape[1:])
    
    if not isinstance(G1, Id):
        orbit_1 = G1.orbits()
        C = torch.einsum("ij,i...->j...", orbit_1, C)
    C = C.transpose(0, 1)

    if not isinstance(G2, Id):
        orbit_2 = G2.orbits()
        C = torch.einsum("ij,i...->j...", orbit_2, C)
    C = C.transpose(0, 1)

    return C


def backward_orbit(grad_C: torch.Tensor, G1: Grammar, G2: Grammar, option: Optional[str] = 'grad') -> torch.Tensor:
    """
    Compute the gradient with respect to the off-diagonals in $G1 \oplus G2$.
    """
    grad_coeff = grad_C

    if not isinstance(G1, Id):
        orbit_1 = G1.orbits()
        if option == 'grad':
            grad_coeff = torch.einsum("ij,j...->i...", orbit_1, grad_coeff)
        elif option == 'coeff':
            grad_coeff = coordinates(grad_coeff, orbit_1)
    grad_coeff = grad_coeff.transpose(0, 1)

    if not isinstance(G2, Id):
        orbit_2 = G2.orbits()
        if option == 'grad':
            grad_coeff = torch.einsum("ij,j...->i...", orbit_2, grad_coeff)
        elif option == 'coeff':
            grad_coeff = coordinates(grad_coeff, orbit_2)
    grad_coeff = grad_coeff.transpose(0, 1)

    return grad_coeff.flatten(0, 1)