from __future__ import annotations
import torch
from utils.grammar import *
from typing import Union


class Group(object):
    """
    Class of permutation groups defined by group(G, \sigma).
    """

    def __init__(self, grammar: Grammar, perm: torch.Tensor, proj_period: Optional[Union[int, float]] = float('inf'), proj_lr: Optional[float] = 0.0, rtol: Optional[float] = None):
        self.grammar = grammar
        self.perm = perm
        self.proj_period = proj_period
        self.proj_lr = proj_lr
        self.rtol = rtol

    def __str__(self):
        return self.grammar.__str__()

    def proj(self, C: torch.Tensor) -> torch.Tensor:
        """
        Project the given matrix C.
        """
        grammar = self.grammar
        if isinstance(grammar, Id):
            return C

        inverse_perm = perm_inverse(self.perm)
        C_perm = C[inverse_perm][:,inverse_perm]
        if grammar.n_basis() > 500:
            C_perm = grammar._proj(C_perm)
        else:
            basis = grammar.basis()
            C_perm = torch.einsum("i,i...->...", coordinates(C_perm, basis), basis)

        return C_perm[self.perm][:,self.perm]

    def proj_orbit(self, C: torch.Tensor) -> torch.Tensor:
        """
        Project the given matrix C with orbits.
        """
        grammar = self.grammar
        if isinstance(grammar, Id):
            return C
        
        inverse_perm = perm_inverse(self.perm)
        orbits = self.grammar.orbits()
        C_perm = C[inverse_perm]
        C_perm = torch.einsum("i,i...->...", coordinates(C_perm, orbits), orbits)
        return C_perm[self.perm]
       
    def proj_S(self, S: torch.Tensor) -> torch.Tensor:
        """
        Return the projected layer of S (which also includes the top column).
        """
        S = S.detach().cpu()
        C = S @ S.t()

        new_C = self.proj_C(C)
        U, D, Vh = torch.svd(new_C)
        new_S = U @ torch.sqrt(torch.diag(D))

        return new_S 
    
    def proj_C(self, C: torch.Tensor) -> torch.Tensor:
        """
        Return the projected layer of C (which also includes the top column).
        """
        n = len(C) - 1
        C = C.detach().cpu()

        center = self.proj(C[1:n+1, 1:n+1])
        upper = self.proj_orbit(C[0, 1:])

        new_C = torch.clone(C)
        new_C[1:n+1, 1:n+1] = center
        new_C[0, 1:] = upper
        new_C[1:, 0] = upper
        new_C = self.proj_lr * new_C + (1 - self.proj_lr) * C

        return new_C

    def proj_error(self, C: torch.Tensor) -> float:
        """
        Return the projection error of C, i.e., ||(proj(C) - C)||_F / ||C||_F .
        """
        if isinstance(self, Id):
            return 0
        else:
            return float(torch.norm(self.proj(C) - C) / torch.norm(C))

    def _forward(self, coeff: torch.Tensor) -> torch.Tensor:
        C = self.grammar._forward(coeff)
        return C[self.perm][:, self.perm]

    def _backward(self, grad_C: torch.Tensor, option: Optional[str] = 'grad') -> torch.Tensor:
        inverse_perm = perm_inverse(self.perm)
        C_perm = grad_C[inverse_perm][:, inverse_perm]
        return self.grammar._backward(C_perm, option)

    def cuda(self) -> Group:
        self.grammar = self.grammar.cuda()
        self.perm = self.perm.cuda()
        return self

    def set_grammar(self, grammar: Grammar) -> Group:
        self.grammar = grammar
        return self

    def set_perm(self, perm: torch.Tensor) -> Group:
        self.perm = perm
        return self