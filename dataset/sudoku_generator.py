from pycryptosat import Solver
import argparse
from itertools import product, combinations, compress
import random
import os
import torch
import torch.nn.functional as F


def tripleToNum(triple, N = 3):
    """return variable index starts from 1"""
    Nsq = N ** 2
    return 1 + triple[0] + triple[1] * Nsq + (triple[2]-1) * Nsq ** 2


def numToTriple(num, N = 3):
    """return triple (i, j, v) for variable index
    i, j starts from 0 and v starts from 1
    """
    Nsq = N ** 2
    if num > Nsq**3: return None
    num -= 1
    triple = []
    for _ in range(3):
        triple.append(num % Nsq)
        num = num // Nsq
    triple[2] += 1
    return tuple(triple)


def get_board(solution, N = 3):
    """convert SAT solution to sudoku solution"""
    Nsq = N ** 2
    board = [[0] * Nsq for _ in range(Nsq)]
    '''
    for (i, j) in product(range(Nsq), repeat = 2):
        for v in range(1, Nsq+1):
            if solution[tripleToNum((i,j,v), N)]: board[i][j] = v
    '''
    for num, ans in enumerate(solution):
        if ans:
            try:
                i, j, v = numToTriple(num, N)
            except:
                pass
            board[i][j] = v
    return board


def get_clause(solution):
    """convert SAT solution to ban clause"""
    return list(compress(range(0, -len(solution), -1), solution))


def ban_solution(solution):
    """naive ban solution clause"""
    return [-i for i,x in enumerate(solution) if x]


def parse(s, N = 3):
    """convert string to board(3x3 only for tests)"""
    Nsq = N ** 2
    board = [[0] * Nsq for _ in range(Nsq)]
    for (i,j) in product(range(Nsq), repeat = 2):
        try: board[i][j] = int(s[Nsq*i+j])
        except: assert s[Nsq*i+j] == '.'
    return board


def unparse(board, N = 3):
    """convert board to string"""
    s = "".join("".join(map(str, e)) for e in board)
    return s.replace("0", ".")
    

def solve(hint, N = 3, check_multiple = False, count_number = False):
    Nsq = N ** 2
    s = Solver(verbose=0)

    # Cell constraints: define, uniqueness
    for (i,j) in product(range(Nsq), repeat = 2):
        s.add_clause([tripleToNum((i,j,v), N) for v in range(1, Nsq+1)])
        for (v1, v2) in combinations(range(1, Nsq+1), 2):
            s.add_clause([-tripleToNum((i,j,v1), N), -tripleToNum((i,j,v2), N)])

    # Row constraints: define, uniqueness
    for (i, v) in product(range(Nsq), range(1, Nsq+1)):
        s.add_clause([tripleToNum((i,j,v), N) for j in range(Nsq)])
        for (j1, j2) in combinations(range(Nsq), 2):
            s.add_clause([-tripleToNum((i,j1,v), N), -tripleToNum((i,j2,v), N)])
    
    # Column constraints: define, uniqueness
    for (j, v) in product(range(Nsq), range(1, Nsq+1)):
        s.add_clause([tripleToNum((i,j,v), N) for i in range(Nsq)])
        for (i1, i2) in combinations(range(Nsq), 2):
            s.add_clause([-tripleToNum((i1,j,v), N), -tripleToNum((i2,j,v), N)])

    # Block constraints: define, uniqueness
    for (i_off, j_off, v) in product(range(0, Nsq, N), range(0, Nsq, N), range(1, Nsq+1)):
        s.add_clause([tripleToNum((i_off+i, j_off+j, v), N) for (i,j) in product(range(N), repeat=2)])
        for ((i1, j1), (i2, j2)) in combinations(product(range(N), repeat = 2), 2):
            s.add_clause([-tripleToNum((i_off+i1, j_off+j1, v), N), -tripleToNum((i_off+i2, j_off+j2, v), N)])

    # Hints
    for (i,j) in product(range(Nsq), repeat = 2):
        v = hint[i][j]
        if (v == 0): continue
        s.add_clause([tripleToNum((i,j,v), N)])
    sat, solution = s.solve()
    
    # Check multiple solutions
    if not sat: raise Exception("No solutions")
    s.add_clause(get_clause(solution))
    sat, new_solution = s.solve()
    if sat: 
        if count_number:
            count = 2
            while True:
                s.add_clause(get_clause(new_solution))
                sat, new_solution = s.solve()
                if not sat:
                    break
                count += 1
            return count
        raise Exception("Multiple solutions")
    return get_board(solution, N)


def generate(N = 3):
    """return a sudoku board with the solution"""
    Nsq = N ** 2
    s = Solver(verbose=0)

    # Cell constraints: define, uniqueness
    for (i,j) in product(range(Nsq), repeat = 2):
        s.add_clause([tripleToNum((i,j,v), N) for v in range(1, Nsq+1)])
        for (v1, v2) in combinations(range(1, Nsq+1), 2):
            s.add_clause([-tripleToNum((i,j,v1), N), -tripleToNum((i,j,v2), N)])

    # Row constraints: define, uniqueness
    for (i, v) in product(range(Nsq), range(1, Nsq+1)):
        s.add_clause([tripleToNum((i,j,v), N) for j in range(Nsq)])
        for (j1, j2) in combinations(range(Nsq), 2):
            s.add_clause([-tripleToNum((i,j1,v), N), -tripleToNum((i,j2,v), N)])
    
    # Column constraints: define, uniqueness
    for (j, v) in product(range(Nsq), range(1, Nsq+1)):
        s.add_clause([tripleToNum((i,j,v), N) for i in range(Nsq)])
        for (i1, i2) in combinations(range(Nsq), 2):
            s.add_clause([-tripleToNum((i1,j,v), N), -tripleToNum((i2,j,v), N)])

    # Block constraints: define, uniqueness
    for (i_off, j_off, v) in product(range(0, Nsq, N), range(0, Nsq, N), range(1, Nsq+1)):
        s.add_clause([tripleToNum((i_off+i, j_off+j, v), N) for (i,j) in product(range(N), repeat=2)])
        for ((i1, j1), (i2, j2)) in combinations(product(range(N), repeat = 2), 2):
            s.add_clause([-tripleToNum((i_off+i1, j_off+j1, v), N), -tripleToNum((i_off+i2, j_off+j2, v), N)])
    
    board = [[0] * Nsq for _ in range(Nsq)]
    assign = {(i,j): set(range(1, Nsq+1)) for (i,j) in product(range(Nsq),repeat=2)}
    tmpvar = Nsq**3+1
    clearlist = []
    while True:
        (i, j) = random.choice(list(assign.keys()))
        v = random.choice(list(assign[(i,j)]))
        sat, solution = s.solve([tripleToNum((i,j,v), N)]+clearlist)
        if sat == False:
            assign[(i,j)].remove(v)
            continue
        s.add_clause([tripleToNum((i,j,v), N)])
        board[i][j] = v
        assign.pop((i,j))
        s.add_clause(get_clause(solution)+[tmpvar])
        sat, new_solution = s.solve(clearlist+[-tmpvar])
        clearlist.append(tmpvar)
        tmpvar += 1
        if sat == False: break
    return board, get_board(solution, N)


def unit_test(input, output, N = 3):
    Nsq = N ** 2
    assert(len(input) == Nsq ** 2)
    return parse(output,N=N) == solve(parse(input, N), N)


def multi_test(input, e, number = None, N = 3):
    Nsq = N ** 2
    assert(len(input) == Nsq ** 2)
    try: 
        if number:
            return number == solve(parse(input, N), N, count_number=True)
        solve(parse(input), N, check_multiple=True)
        return False
    except Exception as msg:
        return (e == str(msg))


def test():
    assert(unit_test("974236158638591742125487936316754289742918563589362417867125394253649871491873625", "974236158638591742125487936316754289742918563589362417867125394253649871491873625"))
    assert(unit_test("2564891733746159829817234565932748617128.6549468591327635147298127958634849362715", "256489173374615982981723456593274861712836549468591327635147298127958634849362715"))
    assert(unit_test("3.542.81.4879.15.6.29.5637485.793.416132.8957.74.6528.2413.9.655.867.192.965124.8", "365427819487931526129856374852793641613248957974165283241389765538674192796512438"))
    assert(unit_test("..2.3...8.....8....31.2.....6..5.27..1.....5.2.4.6..31....8.6.5.......13..531.4..", "672435198549178362831629547368951274917243856254867931193784625486592713725316489"))
    assert(multi_test(".................................................................................", "Multiple solutions"))
    assert(multi_test("........................................1........................................", "Multiple solutions"))
    assert(multi_test("...........5....9...4....1.2....3.5....7.....438...2......9.....1.4...6..........", "Multiple solutions"))
    assert(multi_test("..9.7...5..21..9..1...28....7...5..1..851.....5....3.......3..68........21.....87", "No solutions"))
    assert(multi_test("6.159.....9..1............4.7.314..6.24.....5..3....1...6.....3...9.2.4......16..", "No solutions"))
    assert(multi_test(".4.1..35.............2.5......4.89..26.....12.5.3....7..4...16.6....7....1..8..2.", "No solutions"))
    assert(multi_test("..9.287..8.6..4..5..3.....46.........2.71345.........23.....5..9..4..8.7..125.3..", "No solutions"))
    assert(multi_test(".9.3....1....8..46......8..4.5.6..3...32756...6..1.9.4..1......58..2....2....7.6.", "No solutions"))
    assert(multi_test("....41....6.....2...2......32.6.........5..417.......2......23..48......5.1..2...", "No solutions"))
    assert(multi_test("9..1....4.14.3.8....3....9....7.8..18....3..........3..21....7...9.4.5..5...16..3", "No solutions"))
    assert(multi_test(".39...12....9.7...8..4.1..6.42...79...........91...54.5..1.9..3...8.5....14...87.", "Multiple solutions", number = 2))
    assert(multi_test("..3.....6...98..2.9426..7..45...6............1.9.5.47.....25.4.6...785...........", "Multiple solutions", number = 3))
    assert(multi_test("....9....6..4.7..8.4.812.3.7.......5..4...9..5..371..4.5..6..4.2.17.85.9.........", "Multiple solutions", number = 4))
    assert(multi_test("59.....486.8...3.7...2.1.......4.....753.698.....9.......8.3...2.6...7.934.....65", "Multiple solutions", number = 10))
    assert(multi_test("...3165..8..5..1...1.89724.9.1.85.2....9.1....4.263..1.5.....1.1..4.9..2..61.8...", "Multiple solutions", number = 125))


def test_torch(X, Y, N = 3):
    Nsq = N ** 2
    for i in range(X.size()[0]):
        x = X[i]
        y = Y[i]
        hint = [[0]*Nsq for _ in range(Nsq)]
        board = [[0]*Nsq for _ in range(Nsq)]
        for (i,j) in product(range(Nsq), repeat=2):
            tmp = torch.nonzero(x[i][j])
            hint[i][j] = 0 if tmp.nelement() == 0 else int(tmp+1)
            board[i][j] = int(torch.nonzero(y[i][j])+1)
        hint, board = unparse(hint), unparse(board)
        assert unit_test(hint, board)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_dir', type=str, default='sudoku')
    parser.add_argument('--boardSz', type=int, default=3)
    parser.add_argument('--nRepeat', type=int, default=100)
    parser.add_argument('--check', action='store_true', default=True)
    args = parser.parse_args()

    if args.check:
        test()

    if not os.path.isdir(args.save_dir): 
        os.makedirs(args.save_dir)

    X = torch.zeros(args.nRepeat, args.boardSz**2, args.boardSz**2, args.boardSz**2)
    Y = torch.zeros(args.nRepeat, args.boardSz**2, args.boardSz**2, args.boardSz**2)

    for i in range(args.nRepeat):
        hint, board = generate(args.boardSz)
        x = F.one_hot(torch.tensor(hint), num_classes=10)[:,:,1:]
        y = F.one_hot(torch.tensor(board), num_classes=10)[:,:,1:]
        X[i] = x
        Y[i] = y
    
    if args.check:
        test_torch(X, Y)

    torch.save(X, os.path.join(args.save_dir, 'features.pt'))
    torch.save(Y, os.path.join(args.save_dir, 'labels.pt'))


if __name__=='__main__':
    main()