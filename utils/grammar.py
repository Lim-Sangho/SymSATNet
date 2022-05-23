from __future__ import annotations
from abc import ABCMeta, abstractmethod
from itertools import product
from dataset.cube_generator import Cube333
from utils.algebra import *
import torch


class Grammar(metaclass = ABCMeta):
    """
    Grammar of group symmetries.
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

    def permute(self, perm: torch.Tensor) -> Grammar:
        """
        Permute the basis and orbits.
        Return itself.
        """
        self.perm = perm
        self.basis_cached = self.basis()[:,perm][:,:,perm]
        self.orbits_cached = self.orbits()[:,perm]

        return self
        
    def proj(self, C: torch.Tensor) -> torch.Tensor:
        """
        Return the projected matrix of the layer.
        """
        if isinstance(self, Id):
            return C
        else:
            basis = self.basis()
            return torch.einsum("i,i...->...", coordinates(C, basis), basis)

    def proj_S(self, S: torch.Tensor, proj_lr: float = 1.0) -> torch.Tensor:
        """
        Return the projected layer of S (which also includes the top column).
        """
        S = S.detach().cpu()
        C = S @ S.t()

        new_C = self.proj_C(C, proj_lr)
        U, D, Vh = torch.svd(new_C)
        new_S = U @ torch.sqrt(torch.diag(D))

        return new_S            

    def proj_C(self, C: torch.Tensor, proj_lr: float = 1.0) -> torch.Tensor:
        """
        Return the projected layer of C (which also includes the top column).
        """
        n = len(C) - 1
        C = C.detach().cpu()

        new_C = torch.clone(C)
        top = proj_orbit(C[0:1, 1:], Id(dim = 1), self)[0]
        # top = self.proj(C[0, 1:], self.orbits())
        new_C[0, 1:] = top
        new_C[1:, 0] = top
        new_C[1:n+1, 1:n+1] = self.proj(C[1:n+1, 1:n+1])
        new_C = proj_lr * new_C + (1 - proj_lr) * C

        return new_C

    def proj_error(self, C: torch.Tensor) -> float:
        """
        Return the projection error of C, i.e., ||(proj(C) - C)||_F / ||C||_F .
        """
        if isinstance(self, Id):
            return 0
        else:
            return float(torch.norm(self.proj(C) - C) / torch.norm(C))


class Id(Grammar):
    """
    Class of identity groups.
    """
    
    def __init__(self, dim: int) -> None:
        self.dim = dim
        self.perm = torch.arange(0, dim)
        self.basis_cached = None
        self.orbits_cached = None

    def __str__(self) -> str:
        return f"I{self.dim}"

    def _proj(self, C: torch.Tensor) -> torch.Tensor:
        return C

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
        self.perm = torch.arange(0, dim)
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

    def basis(self) -> torch.Tensor:
        if self.basis_cached is None:
            identity = torch.eye(self.dim)
            basis = []
            for i in range(self.dim):
                basis.append(torch.roll(identity, i, 0))
            self.basis_cached = torch.stack(basis)

        return self.basis_cached

    def orbits(self) -> torch.Tensor:
        if self.orbits_cached is None:
            self.orbits_cached = torch.ones(1, self.dim)
        return self.orbits_cached

    def n_basis(self) -> int:
        return self.dim

    def n_orbits(self) -> int:
        return 1
        

class Perm(Grammar):
    """
    Class of permutation groups.
    """

    def __init__(self, dim: int) -> None:
        self.dim = dim
        self.perm = torch.arange(0, dim)
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

    def __init__(self, dim: int, perms: torch.Tensor) -> None:
        assert perms.shape[1] == dim
        self.dim = dim
        self.perm = torch.arange(0, dim)
        self.perms = perms.long()
        self.basis_cached = None
        self.orbits_cached = None

    def __str__(self) -> str:
        return f"<{self.perms.tolist()}>"

    def _proj(self, C: torch.Tensor) -> torch.Tensor:
        vector_inner = torch.einsum("ijk,jk...->i...", self.basis(), C)
        basis_inner = torch.einsum("i...->i", self.basis())
        coeffs = torch.einsum("i...,i->i...", vector_inner, 1 / basis_inner)
        C_proj = torch.einsum("ijk,i...->jk...", self.basis(), coeffs)

        return C_proj

    def basis(self) -> torch.Tensor:
        if self.basis_cached is None:
            I = torch.eye(self.dim)
            rho = torch.cat([torch.kron(I[perm], I[perm]) - torch.kron(I, I) for perm in self.perms])
            U, S, V = torch.svd(rho)
            is_null = torch.isclose(S, torch.zeros(1), atol = torch.norm(rho) * 1e-6)
            basis = V.T[is_null].reshape(-1, self.dim, self.dim)
            self.basis_cached = basis

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
                    for e, perm in product(orbit, self.perms):
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
        self.perm = torch.arange(0, self.dim)
        self.basis_cached = None
        self.orbits_cached = None

    def __str__(self) -> str:
        return f"({self.G1} \u2295 {self.G2})"

    def _proj(self, C: torch.Tensor) -> torch.Tensor:
        dim_1 = self.G1.dim
        C_proj = torch.zeros(C.shape)
        C_proj[:dim_1, :dim_1] = self.G1._proj(C[:dim_1, :dim_1])
        C_proj[dim_1:, dim_1:] = self.G2._proj(C[dim_1:, dim_1:])

        upper = C[:dim_1, dim_1:]
        lower = C[dim_1:, :dim_1]

        upper_proj = proj_orbit(upper, self.G1, self.G2)
        lower_proj = proj_orbit(lower, self.G2, self.G1)

        C_proj[:dim_1, dim_1:] = upper_proj
        C_proj[dim_1:, :dim_1] = lower_proj

        return C_proj

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
        self.perm = torch.arange(0, self.dim)
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
        self.perm = torch.arange(0, self.dim)
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
        return self.G1.n_basis() * self.G2.n_orbits() + self.G2.n_basis() - self.G2.n_orbits()

    def n_orbits(self) -> int:
        return self.G1.n_orbits() * self.G2.n_orbits()


class Sudoku(Kron):
    """
    Class of the Sudoku group.
    """

    def __init__(self) -> None:
        perm_row_column = Wreath(Perm(dim = 3), Perm(dim = 3))
        perm_number = Perm(dim = 9)
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
        perm_move = Gen(dim = 54, perms = perm_move)

        perm_color = torch.Tensor([[4, 0, 2, 3, 5, 1], [2, 1, 5, 0, 4, 3], [0, 2, 4, 1, 3, 5]])
        perm_color = Gen(dim = 6, perms = perm_color)

        super().__init__(perm_move, perm_color)

        self.perm_move = perm_move
        self.perm_color = perm_color