from __future__ import annotations
import torch
import math


def divisors(n: int) -> list[int]:
    """
    Find divisors of the natural number.
    """
    divs = [x for x in range(1, int(math.sqrt(n)) + 1) if n % x == 0]
    opps = [n // x for x in reversed(divs) if x ** 2 != n]
    return divs + opps


def coordinates(vector: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    """
    Find the coordinates of the vector correspoding to the basis.
    """
    summand_letters = "bcdefghijklmnopqrstuvwxyz"[:basis.dim()-1]
    basis_inner = torch.einsum("a"+summand_letters+"->a", basis * basis)
    vector_inner = torch.einsum("a"+summand_letters+","+summand_letters+"...->a...", basis, vector)
    return torch.einsum("a,a...->a...", 1 / basis_inner, vector_inner)


def gram_schmidt(vectors: torch.Tensor) -> torch.Tensor:
    """
    Do Gram Schmidt process with the linearly independent row vectors.
    """
    vectors = vectors.clone()
    for i in range(len(vectors)):
        vector = vectors[i]
        for j in range(i):
            if torch.allclose(vectors[j], torch.zeros(1)):
                vector = vector
            else:
                vector = vector - (torch.sum(vector * vectors[j]) / torch.sum(vectors[j] * vectors[j])) * vectors[j]
        vectors[i] = vector
        
    is_zero = torch.Tensor([torch.allclose(vector, torch.zeros(1)) for vector in vectors])
    is_nonzero = torch.logical_not(is_zero)
    vectors = vectors[is_nonzero]
    vectors = torch.nn.functional.normalize(vectors, dim = 1)
    
    return vectors


def swap(tensor: torch.Tensor, i: int, j: int) -> torch.Tensor:
    """
    Swap the indices i and j of tensor.
    """
    dim = tensor.dim()
    shift = tuple(range(1, dim)) + (0,)
    perm = torch.arange(len(tensor))
    perm[i], perm[j] = j, i

    for _ in range(dim):
        tensor = tensor[perm]
        tensor = tensor.permute(shift)

    return tensor


def shuffle(vector: torch.Tensor, n_shuffle: int) -> torch.Tensor:
    """
    Shuffle the indices of the vector.
    """
    n = len(vector)
    for _ in range(n_shuffle):
        i = torch.randint(0, n, (1,))
        j = torch.randint(0, n, (1,))
        vector = swap(vector, i, j)
    return vector


def set_distance(A: torch.Tensor, B: torch.Tensor) -> float:
    """
    Compute the set distance between the vectors A and B.
    We use the Hausdorff distance,
    i.e., the maximum between $sup_{a \in A} inf_{b \in B} |a - b|$
    and $sup_{b \in B} inf_{a \in A} |a - b|$.
    """
    inf_a = torch.stack([torch.min(torch.abs(a - B)) for a in A.flatten()])
    inf_b = torch.stack([torch.min(torch.abs(A - b)) for b in B.flatten()])
    sup_a = torch.max(inf_b)
    sup_b = torch.max(inf_a)

    return float(max(sup_a, sup_b))


def normalize(vector: torch.Tensor) -> torch.Tensor:
    """
    Normalize the vector.
    """
    return (vector - torch.mean(vector)) / torch.std(vector, unbiased = False)


def perm_inverse(perm: torch.Tensor) -> torch.Tensor:
    """
    Find the inverse permutation of perm.
    """
    inversed = []
    for i in range(len(perm)):
        where = torch.where(perm == i)[0]
        inversed.append(where)
    inversed = torch.cat(inversed)
    return inversed


def perm_sum(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """
    Find the direct sum of permutations A and B.
    """
    return torch.cat([A, B + len(A)])


def perm_kron(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """
    Find the Kronecker product of permutations A and B.
    """
    return torch.arange(len(A) * len(B)).reshape(len(A), len(B))[A][:,B].flatten()


def perm_wreath(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """
    Find the wreath product of permutations A and B.
    """
    return perm_kron(B, A)