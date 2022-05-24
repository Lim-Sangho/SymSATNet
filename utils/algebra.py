from __future__ import annotations
from typing import TYPE_CHECKING, Optional
from sklearn.cluster import KMeans
import torch
import math

if TYPE_CHECKING:
    from utils.grammar import Grammar


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
    basis_inner = torch.einsum("i...->i", basis * basis)
    vector_inner = torch.einsum("i...->i", basis * vector)
    return vector_inner / basis_inner


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

    
# def reduce_basis(basis: torch.Tensor) -> torch.Tensor:
#     """
#     Remove linearly dependent vectors in the basis.
#     """
#     basis = gram_schmidt(basis)
#     is_zero = torch.Tensor([torch.allclose(B, torch.zeros(1)) for B in basis])
#     is_nonzero = torch.logical_not(is_zero)
#     return basis[is_nonzero]


# def clustering(vectors: torch.Tensor, n_clusters: int, tol: float) -> torch.Tensor:
#     """
#     Do clustering with the row vectors.
#     """
#     labels = - torch.ones(len(vectors))
#     for n in range(n_clusters):
#         index = min(torch.nonzero(labels == -1).flatten().tolist(), default = None)
#         if index is None:
#             break
#         for i in range(len(vectors)):
#             threshold = torch.norm(vectors[index]) * tol
#             if torch.norm(vectors[i] - vectors[index]) <= threshold:
#                 labels[i] = n
    
#     return labels


def swap(tensor: torch.Tensor, i: int, j: int) -> torch.Tensor:
    """
    Swap the indices i and j of tensor.
    """
    # n = len(vector)

    # if vector.dim() == 1:
    #     new_vector = vector.clone()
    #     new_vector[i], new_vector[j] = vector[j], vector[i]
    #     return new_vector

    # if vector.dim() == 2:
    #     perm = torch.arange(0, n)
    #     perm[i], perm[j] = j, i
    #     perm = torch.eye(n)[perm]
    #     return perm @ vector @ perm
    
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

def shuffle_all(vectors: torch.Tensor, n_shuffle: int) -> torch.Tensor:
    """
    Shuffle the indices of all the vectors with the same permutation.
    """
    n = vectors.shape[1]
    new_vectors = vectors.clone()
    for _ in range(n_shuffle):
        i = torch.randint(0, n, (1,))
        j = torch.randint(0, n, (1,))
        for k in range(len(new_vectors)):
            new_vectors[k] = swap(new_vectors[k], i, j)
    return new_vectors


def kmeans_auto(vectors: torch.Tensor, rtol: Optional[float] = 1) -> KMeans:
    """
    Do clustering with n_clusters such that
    inertia(n_clusters) < atol and inertia(n_clusters - 1) >= atol.
    """
    assert len(vectors > 0)
    if vectors.dim() == 1:
        vectors = torch.stack([vectors, vectors]).T

    upper_bound = len(vectors)
    lower_bound = 1
    atol = torch.var(vectors) * rtol
    cache = {}

    while upper_bound > lower_bound:
        n_clusters = (upper_bound + lower_bound) // 2
        kmeans = KMeans(n_clusters = n_clusters)
        kmeans.fit(vectors)
        cache[n_clusters] = kmeans
        inertia = kmeans.inertia_
        if inertia < atol:
            upper_bound = n_clusters
        else:
            lower_bound = n_clusters + 1
    
    if upper_bound not in cache:
        kmeans = KMeans(n_clusters = upper_bound)
        kmeans.fit(vectors)
        cache[upper_bound] = kmeans
    
    return cache[upper_bound]


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