from __future__ import annotations
from typing import Optional
from functools import reduce
from itertools import compress
from utils.algebra import *
from utils.grammar import *
from utils.group import *
import numpy as np
import torch


def prodfind(C: torch.Tensor, rtol: float, max_err: float) -> list[tuple[Grammar, torch.Tensor]]:
    """
    Split the matrix into a sum of Kronecker products, i.e., $C = \sum_i {A_i \otimes B_i}$.
    """
    N = C.shape[0]
    splits = []

    for m in reversed(divisors(N)):
        if m > 1 and m < N // 1:
            n = N // m
            C_hat = []
            for j in range(m):
                for i in range(m):
                    C_ij = C[i*n:(i+1)*n, j*n:(j+1)*n]
                    C_hat.append(C_ij.T.flatten())
            C_hat = torch.stack(C_hat)
            U, S, V = torch.svd(C_hat)

            cut = torch.where(S > (S[0] / 5))[0]
            rank = max(cut, default = 0) + 1

            # print(m)
            # print(rank)
            # print(S)
            # print(grad)
            # print(S[0] / 5)
            # plt.plot(S)
            # plt.show()
            # print()

            if rank <= 6:
                S = S[:rank]
                A = U[:,:rank]
                B = V[:,:rank]
                A = A * torch.sqrt(S)
                B = B * torch.sqrt(S)
                A = A.T.reshape(-1, m, m).transpose(1, 2)
                B = B.T.reshape(-1, n, n).transpose(1, 2)
                
                splits.append((A, B))

    return [wreathfind_in_prodfind(A, B, rtol, max_err) for A, B in splits]


def wreathfind_in_prodfind(A: torch.Tensor, B: torch.Tensor, rtol: float, max_err: float) -> tuple[Grammar, torch.Tensor]:
    """
    Check whether the split of Kronecker product is a wreath product.
    """
    if not len(A) == len(B) == 2:
        G_1, perm_1 = symfind(A[0], rtol, max_err)
        G_2, perm_2 = symfind(B[0], rtol, max_err)
        return Kron(G_1, G_2), perm_kron(perm_1, perm_2)
    
    eye = torch.eye(A.shape[1])
    ones = torch.ones(B.shape[1], B.shape[2])
    coeff_A = coordinates(eye, A)
    coeff_B = coordinates(ones, B)
    atol = rtol * 0.4

    # print(coeff_A * coeff_B)
    # print(torch.abs(torch.sum(coeff_A * coeff_B)))
    # print(atol)

    if torch.abs(torch.dot(coeff_A, coeff_B)) > atol:
        G_1, perm_1 = symfind(A[0], rtol, max_err)
        G_2, perm_2 = symfind(B[0], rtol, max_err)
        return Kron(G_1, G_2), perm_kron(perm_1, perm_2)

    else:
        alpha = coeff_B[0]
        beta = coeff_A[1]
        A = torch.stack([A[0] / alpha, eye])
        B = torch.stack([ones, B[1] / beta])
        G_1, perm_1 = symfind(B[1], rtol, max_err)
        G_2, perm_2 = symfind(A[0], rtol, max_err)
        return Wreath(G_1, G_2), perm_wreath(perm_1, perm_2)


def sumfind(C: torch.Tensor, rtol: float, max_err: float) -> tuple[Grammar, torch.Tensor]:
    """
    Split the matrix into a direct sum, i.e., $C = \bigoplus_i B_i$.
    """
    n_repeat = 10
    N = C.shape[0]
    C, perm, blocks = find_blocks(C, rtol)

    for _ in range(n_repeat):
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

    return wreathfind_in_sumfind(C, perm, blocks, super_blocks, rtol, max_err)


def wreathfind_in_sumfind(C: torch.Tensor, perm: torch.Tensor,
                 blocks: torch.Tensor, super_blocks: torch.Tensor,
                 rtol: float, max_err: float) -> tuple[Grammar, torch.Tensor]:
    N = C.shape[0]
    summands = []
    perms = []
    num_blocks = []

    for i in range(len(super_blocks)):
        start_i = super_blocks[i]
        end_i = super_blocks[i+1] if i < len(super_blocks) - 1 else N
        num_blocks.append(torch.sum(torch.logical_and(blocks >= start_i, blocks < end_i)))
    num_blocks = torch.Tensor(num_blocks).long()

    if len(super_blocks) == 1 and num_blocks[0] == 1:
        Sn = Symm(dim = N)
        if Group(Sn, torch.arange(N)).proj_error(C) <= rtol:
            return Symm(dim = N), perm_inverse(perm)

        Zn = Cyclic(dim = N)
        if Group(Zn, torch.arange(N)).proj_error(C) <= rtol:
            return Cyclic(dim = N), perm_inverse(perm)

        return Id(dim = N), torch.arange(0, N)

    for i in range(len(super_blocks)):
        start_i = super_blocks[i]
        end_i = super_blocks[i+1] if i < len(super_blocks) - 1 else N
        B_i = C[start_i:end_i, start_i:end_i]
        m = num_blocks[i]
        n = len(B_i) // m

        if m == 1:
            G_i, perm_i = symfind(B_i, rtol, max_err)
            summands.append(G_i)
            perms.append(perm_i)

        elif n == 1:
            summands.append(Id(dim = len(B_i)))
            perms.append(torch.arange(0, len(B_i)))

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

            G_i1, perm_i1 = symfind(A, rtol, max_err)
            G_i2, perm_i2 = symfind(B, rtol, max_err)
            summands.append(Wreath(G_i1, G_i2))
            perms.append(perm_wreath(perm_i1, perm_i2))

    return reduce(Sum, summands), reduce(perm_sum, perms)[perm_inverse(perm)]


def find_blocks(C: torch.Tensor, rtol: float) -> Optional[tuple[torch.Tensor]]:
    """
    Find the blocks of the layer C.
    """
    atol = torch.norm(C) * rtol
    N = C.shape[0]
    i = 0
    j = 0
    block = 0
    blocks = {0}
    C_prev = C.clone()
    perm = torch.arange(0, N)
    perm_prev = perm.clone()
    argsort = None
    
    while i < N-1:

        if block == i:
            blocks.add(block)
            diags = torch.diag(C)[i:]
            diag_close = torch.nonzero(torch.abs(diags - diags[0]) <= atol * 0.6).flatten()[1:] + i

            if len(diag_close) > 0:
                if argsort is None:
                    argsort = np.argsort([np.isclose(C[i, diag_close], pixel, atol = atol * 0.4).sum() for pixel in C[i, diag_close]])
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

            # draw(C)
            # draw(torch.cat([chunk_i.reshape(1, -1), chunks], dim = 0))
            # print(distances_chunk)
            # print(atol)
            # print(torch.min(distances_chunk) <= atol)

            if torch.min(distances_chunk) <= atol * 0.3:
                index = torch.argmin(distances_chunk).item() + (i+1)
                C = swap(C, i+1, index)
                perm = swap(perm, i+1, index)
                i += 1

            else:
                off_diag_row = C[block:i+1, i+1:]
                off_diag_col = C[i+1:, block:i+1]
                off_diag = torch.cat([off_diag_row, off_diag_col.T], dim = 1)
                distance_off_diag = torch.max(torch.abs(off_diag - off_diag[0]))

                # draw(C)
                # draw(off_diag)
                # print(distance_off_diag)
                # print(atol)
        
                if distance_off_diag <= atol * 0.3:
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
    Merge the similar blocks into super blocks.
    Each super block constructs a wreath product.
    """
    atol = torch.norm(C) * 0.4 * rtol
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


def max_group(C: torch.Tensor, groups: list[tuple[Grammar, torch.Tensor]], max_err: float) -> tuple[Grammar, torch.Tensor]:
    """
    Find the maximum group which contains the minimum number of basis elements among the groups.
    """
    N = C.shape[0]

    if len(groups) == 0:
        return Id(dim = N), torch.arange(0, N)

    if len(groups) == 1:
        return groups[0]

    else:
        print(*[grammar for grammar, perm in groups])
        print(*[grammar.n_basis() for grammar, perm in groups])
        print(*[Group(grammar, perm).proj_error(C) for grammar, perm in groups])
        print()

        errors = torch.Tensor([Group(grammar, perm).proj_error(C) for grammar, perm in groups])
        error_filter = errors <= max_err
        
        if not torch.any(error_filter):
            return Id(dim = N), torch.arange(0, N)

        groups = list(compress(groups, error_filter))
        errors = errors[error_filter]

        n_basis = torch.Tensor([grammar.n_basis() for grammar, perm in groups])
        groups = list(compress(groups, n_basis == n_basis.min()))
        errors = errors[n_basis == n_basis.min()]

        return groups[errors.argmin()]


def symfind(C: torch.Tensor, rtol: float, max_err: float) -> tuple[Grammar, torch.Tensor]:
    """
    Find the group symmetry of the matrix C.
    """
    assert C.shape[0] == C.shape[1]
    N = C.shape[0]
    C = normalize(C)
    groups = []

    Sn = Symm(dim = N)
    if Group(Sn, torch.arange(N)).proj_error(C) <= rtol:
        return Sn, torch.arange(0, N)

    Zn = Cyclic(dim = N)
    if Group(Zn, torch.arange(N)).proj_error(C) <= rtol:
        groups.append((Zn, torch.arange(0, N)))
        
    if N < 70:
        groups.append(sumfind(C, rtol, max_err))
    groups.extend(prodfind(C, rtol, max_err))

    return max_group(C, groups, max_err)