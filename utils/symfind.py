from __future__ import annotations
from typing import Iterable, Optional
from functools import reduce
from sklearn.cluster import KMeans
from torch import threshold
from utils.algebra import *
from utils.grammar import *
from utils.draw import draw
import matplotlib.pyplot as plt
import numpy as np
import torch
import math


def kron_split(C: torch.Tensor, rtol: float) -> list[tuple[torch.Tensor]]:
    """
    Split the layer into a sum of Kronecker products, i.e., $C = \sum_i {A_i \otimes B_i}$.
    If a sum of Kronecker products is not found, return None.
    """
    N = C.shape[0]
    thresh_rank = 6
    thresh_ratio = 1
    # thresh_ratio = 0.25
    splits = []

    for m in reversed(divisors(N)):
        if m <= 1 or m >= N // 1:
            continue
        n = N // m

        C_hat = []
        for j in range(m):
            for i in range(m):
                C_ij = C[i*n:(i+1)*n, j*n:(j+1)*n]
                C_hat.append(C_ij.T.flatten())
        C_hat = torch.stack(C_hat)
        U, S, V = torch.svd(C_hat)

        atol = S[0] / 5
        # grad = torch.Tensor([S[i] - S[i+1] for i in range(len(S)-1)])
        # cut = torch.where(grad > atol)[0]
        cut = torch.where(S > atol)[0]
        rank = max(cut, default = 0) + 1
        ratio = rank / len(S)

        # print(m)
        # print(rank)
        # print(ratio)
        # print(S)
        # print(grad)
        # print(atol)
        # plt.plot(S)
        # plt.show()
        # print()

        if rank <= thresh_rank and ratio <= thresh_ratio:
            S = S[:rank]
            A = U[:,:rank]
            B = V[:,:rank]

            A = A * torch.sqrt(S)
            B = B * torch.sqrt(S)

            A = A.T.reshape(-1, m, m).transpose(1, 2)
            B = B.T.reshape(-1, n, n).transpose(1, 2)

            splits.append((A, B))

    return splits


def sum_split(C: torch.Tensor, rtol: float) -> tuple[torch.Tensor]:
    """
    Split the layer into a direct sum, i.e., $C = \bigoplus_i B_i$.
    If a sum of Kronecker products is not found, return None.
    """
    n_trials = 10
    N = len(C)
    C, perm, blocks = find_blocks(C, rtol)

    for _ in range(n_trials - 1):
        sub_blocks = torch.Tensor([]).long()
        sub_perms = torch.Tensor([]).long()
        for i in range(len(blocks)):
            start_i = blocks[i]
            end_i = blocks[i+1] if i < len(blocks) - 1 else N
            B_i = C[start_i:end_i, start_i:end_i]
            _, perm_i, blocks_i = find_blocks(B_i, rtol)
            sub_blocks = torch.cat([sub_blocks, blocks_i + start_i])
            sub_perms = torch.cat([sub_perms, perm_i + start_i])
            perm[start_i:end_i] = perm[start_i:end_i][perm_i]
        C = C[sub_perms][:,sub_perms]
        blocks = sub_blocks

    C, perm, blocks = sort_blocks(C, perm, blocks)
    C, perm, blocks, super_blocks = merge_blocks(C, perm, blocks, rtol)

    # draw(C)
    # print(blocks)
    # print(super_blocks)

    return C, perm, blocks, super_blocks


def wreath_split(C: torch.Tensor, perm: torch.Tensor,
                 blocks: torch.Tensor, super_blocks: torch.Tensor, rtol: float) -> Grammar:
    N = len(C)
    summands = []
    num_blocks = []

    for i in range(len(super_blocks)):
        start_i = super_blocks[i]
        end_i = super_blocks[i+1] if i < len(super_blocks) - 1 else len(C)
        num_blocks.append(torch.sum(torch.logical_and(blocks >= start_i, blocks < end_i)))
    num_blocks = torch.Tensor(num_blocks).long()

    if len(super_blocks) == 1 and num_blocks[0] == 1:
        Sn = Perm(dim = N)
        Cn = Cyclic(dim = N)
        if Sn.proj_error(C) <= rtol * 2.5:
            return Sn.permute(perm_inverse(perm))
        elif Cn.proj_error(C) <= rtol * 2.5:
            return Cn.permute(perm_inverse(perm))
        else:
            return Id(dim = N)

    for i in range(len(super_blocks)):
        start_i = super_blocks[i]
        end_i = super_blocks[i+1] if i < len(super_blocks) - 1 else N
        B_i = C[start_i:end_i, start_i:end_i]
        m = num_blocks[i]
        n = len(B_i) // m

        if m == 1:
            summands.append(sym_find(B_i, rtol))
        elif n == 1:
            summands.append(Id(dim = len(B_i)))
        else:
            A = torch.zeros(n, n)
            for i in range(m):
                A += B_i[n*i:n*(i+1), n*i:n*(i+1)]
            A /= m

            off_diag = B_i - torch.kron(torch.eye(m), A)
            for i in range(m):
                off_diag[n*i:n*(i+1), n*i:n*(i+1)] += torch.mean(A)

            B = torch.zeros(m, m)
            for i in range(m):
                for j in range(m):
                    B[i, j] = torch.mean(off_diag[n*i:n*(i+1), n*j:n*(j+1)])

            summands.append(Wreath(sym_find(A, rtol), sym_find(B, rtol)))

    return reduce(Sum, summands).permute(perm_inverse(perm))


def find_blocks(C: torch.Tensor, rtol: float) -> Optional[tuple[torch.Tensor]]:
    """
    Find the blocks of the layer C.
    """
    atol = torch.norm(C) * rtol
    N = C.shape[0]
    C_prev = C.clone()
    i = 0
    j = 0
    block = 0
    blocks = {0}
    argsort = None
    perm = torch.arange(0, N)
    perm_prev = perm.clone()
    
    while i < N-1:

        if block == i:
            blocks.add(block)
            diags = torch.diag(C)[i:]
            diag_close = torch.nonzero(torch.abs(diags - diags[0]) <= atol * 1.5).flatten()[1:] + i

            # if len(C) == 3:
            #     draw(C)
            #     print(diag_close)

            if len(diag_close) > 0:
                if argsort is None:
                    # kmeans = kmeans_auto(C[i, diag_close])
                    # labels = kmeans.labels_
                    # argsort = np.argsort([(labels == label).sum() for label in labels])
                    argsort = np.argsort([np.isclose(C[i, diag_close], pixel, atol = atol).sum() for pixel in C[i, diag_close]])
                if j >= len(argsort):
                    i += 1
                    j = 0
                    block = i
                    C_prev = C.clone()
                    perm_prev = perm.clone()
                    argsort = None
                    continue
                index = diag_close[argsort[j]]

                C = swap(C, i+1, index)
                perm = swap(perm, i+1, index)
                i += 1
            
            else:
                i += 1
                block += 1

        else:
            row_i = C[i, block:i]
            col_i = C[block:i, i]
            diag_i = C[i, i]
            chunk_i = torch.cat([row_i, col_i, torch.Tensor([diag_i])])

            rows = C[i+1:, block+1:i+1]
            cols = C[block+1:i+1, i+1:]
            diags = torch.diag(C)[i+1:]
            chunks = torch.cat([rows, cols.T, diags.reshape(-1, 1)], dim = 1)

            distances_chunk = torch.max(torch.abs(chunks - chunk_i), dim = 1)[0]
            # distances_chunk = torch.mean(torch.abs(chunks - chunk_i), dim = 1)
            # distances_chunk = torch.norm(chunks - chunk_i, dim = 1)

            # draw(C)
            # draw(torch.cat([chunk_i.reshape(1, -1), chunks], dim = 0))
            # print(distances_chunk)
            # print(atol)
            # print(torch.min(distances_chunk) <= atol)

            if torch.min(distances_chunk) <= atol * 0.75:
                index = torch.argmin(distances_chunk).item() + (i+1)
                C = swap(C, i+1, index)
                perm = swap(perm, i+1, index)
                i += 1

            else:
                off_diag_row = C[block:i+1, i+1:]
                off_diag_col = C[i+1:, block:i+1]
                off_diag = torch.cat([off_diag_row, off_diag_col.T], dim = 1)
                distance_off_diag = torch.max(torch.abs(off_diag - off_diag[0]))
                # distance_off_diag = torch.max(torch.mean(torch.abs(off_diag - off_diag[0]), dim = 1))
                # distance_off_diag = torch.max(torch.norm(off_diag - off_diag[0], dim = 1))

                # draw(C)
                # draw(off_diag)
                # print(distance_off_diag)
                # print(atol)
        
                if distance_off_diag <= atol * 0.75:
                    i += 1
                    j = 0
                    block = i
                    C_prev = C.clone()
                    perm_prev = perm.clone()
                    argsort = None
                else:
                    i = block
                    j += 1
                    C = C_prev.clone()
                    perm = perm_prev.clone()

    return C, perm, torch.Tensor(sorted(list(blocks))).long()


def sort_blocks(C: torch.Tensor, perm: torch.Tensor, blocks: torch.Tensor) -> tuple[torch.Tensor]:
    """
    Sort the blocks of the layer C.
    Return the splitted blocks and the permutation of indices.
    """
    block_info = []
    for i in range(len(blocks)):
        start_i = blocks[i]
        end_i = blocks[i+1] if i < len(blocks) - 1 else len(C)
        B_i = C[start_i:end_i, start_i:end_i]
        block_info.append([torch.arange(start_i, end_i), B_i])

    block_info.sort(key = lambda x: [len(x[1]), torch.sum(x[1])]) # Sort by (len(B_i), sum(B_i)) with dictionary order.
    block_perm, B = zip(*block_info)
    block_perm = torch.cat(block_perm)
    C = C[block_perm][:,block_perm]
    perm = perm[block_perm]
    blocks = [0]
    for B_i in B[:-1]:
        blocks.append(blocks[-1] + len(B_i))

    return C, perm, torch.Tensor(blocks).long()


def merge_blocks(C: torch.Tensor, perm: torch.Tensor, blocks: torch.Tensor, rtol: float) -> tuple[torch.Tensor]:
    """
    Merge the same blocks in B.
    Each merged block should occur a wreath product.
    """
    atol = torch.norm(C) * rtol
    super_blocks = blocks.clone()
    for i in range(1, len(blocks)):
        start_prev = blocks[i-1]
        start_i = blocks[i]
        end_i = blocks[i+1] if i < len(blocks) - 1 else len(C)
        B_prev = C[start_prev:start_i, start_prev:start_i]
        B_i = C[start_i:end_i, start_i:end_i]
        # draw(B_prev)
        # draw(B_i)
        # print(set_distance(B_prev, B_i))
        # print(atol)
        if len(B_prev) == len(B_i) and set_distance(B_prev, B_i) <= atol:
            B_i, perm_i = perm_block(B_i, B_prev)
            sub_perm = torch.cat([torch.arange(0, start_i), perm_i + start_i, torch.arange(end_i, len(C))])
            C = C[sub_perm][:,sub_perm]
            perm[start_i:end_i] = perm[start_i:end_i][perm_i]
            super_blocks = super_blocks[super_blocks != start_i]

    return C, perm, blocks, super_blocks


def perm_block(A: torch.Tensor, B: torch.Tensor) -> tuple[torch.Tensor]:
    """
    Permute block A to make it close to B.
    """
    perm = torch.arange(len(A))
    for i in range(1, len(A)):
        row_A = A[0, i:]
        row_B = B[0, i:]
        argmin = torch.argmin(torch.abs(row_A - row_B[0]))
        A = swap(A, i, argmin+i)
        perm = swap(perm, i, argmin+i)
    
    return A, perm


def check_wreath(A: torch.Tensor, B: torch.Tensor, rtol: float) -> Optional[tuple[torch.Tensor]]:
    """
    Check whether the split of Kronecker product is a wreath product.
    """
    if not len(A) == len(B) == 2:
        return None
    
    eye = torch.eye(A[0].shape[0])
    ones = torch.ones(B[0].shape)
    coeff_A = coordinates(eye, A)
    coeff_B = coordinates(ones, B)
    atol = rtol

    # print(coeff_A * coeff_B)
    # print(torch.abs(torch.sum(coeff_A * coeff_B)))
    # print(atol)

    if torch.abs(torch.sum(coeff_A * coeff_B)) > atol:
        return None

    else:
        alpha = coeff_B[0]
        beta = coeff_A[1]
        A = torch.stack([A[0] / alpha, eye])
        B = torch.stack([ones, B[1] / beta])
        return A, B


def maximal_grammar(C: torch.Tensor, grammars: list[Grammar]) -> Grammar:
    """
    Find the maximal grammar which has the minimum number of basis vectors among the grammars.
    """
    if len(grammars) == 0:
        return Id(dim = C.shape[0])

    if len(grammars) == 1:
        return grammars[0]

    else:
        # print(*grammars)
        # print(*[grammar.n_basis() for grammar in grammars])
        # print(*[grammar.proj_error(C) for grammar in grammars])
        # print()

        # thresh_error = 0.4 # when corrupt_num = 0
        thresh_error = 0.6 # otherwise
        grammars = [grammar for grammar in grammars if grammar.proj_error(C) < thresh_error]
        if len(grammars) == 0:
            return Id(dim = C.shape[0])

        n_basis = torch.Tensor([grammar.n_basis() for grammar in grammars])
        argmin = torch.where(n_basis == torch.min(n_basis))[0]
        errors = torch.Tensor([grammars[i].proj_error(C) for i in argmin])
        
        return grammars[argmin[torch.argmin(errors)]]


def sym_find(C: torch.Tensor, rtol: float) -> Grammar:
    """
    Find the underlying group structure of the layer.
    The detection priority is determined by the number of basis matrices.
    """
    C = normalize(C)
    assert C.shape[0] == C.shape[1]
    N = C.shape[0]
    grammars = []

    Sn = Perm(dim = N)
    Cn = Cyclic(dim = N)
    if Sn.proj_error(C) <= rtol * 2.5:
        return Sn
    if Cn.proj_error(C) <= rtol * 2.5:
        grammars.append(Cn)

    for A, B in kron_split(C, rtol):
        wreath = check_wreath(A, B, rtol)
        if wreath is not None:
            A, B = wreath
            grammars.append(Wreath(sym_find(B[1], rtol), sym_find(A[0], rtol)))
        else:
            grammars.append(Kron(sym_find(A[0], rtol), sym_find(B[0], rtol)))
    
    if N < 70:
        C_perm, perm, blocks, super_blocks = sum_split(C, rtol)
        grammars.append(wreath_split(C_perm, perm, blocks, super_blocks, rtol))

    return maximal_grammar(C, grammars)