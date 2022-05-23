from __future__ import annotations
from typing import Iterable, Optional
import numpy as np
import torch
import os


class Cube333(object):
    """
    Class of initially solved 3x3x3 cubes.
    """
    
    def __init__(self) -> None:
        self.state = np.array([[i] * 9 for i in range(6)])
        self.index_state = np.arange(54).reshape(6, 9)

        self.corner_indices = [[(1, 0), (3, 8), (5, 6)], [(1, 2), (5, 8), (2, 2)],
                               [(1, 8), (2, 0), (0, 2)], [(1, 6), (0, 0), (3, 6)],
                               [(4, 0), (0, 8), (2, 6)], [(4, 2), (2, 8), (5, 2)],
                               [(4, 8), (5, 0), (3, 2)], [(4, 6), (3, 0), (0, 6)]]
        self.edge_indices = [[(1, 1), (5, 7)], [(1, 5), (2, 1)],
                             [(1, 7), (0, 1)], [(1, 3), (3, 7)],
                             [(2, 3), (0, 5)], [(2, 5), (5, 5)],
                             [(3, 3), (0, 3)], [(3, 5), (5, 3)],
                             [(4, 1), (2, 7)], [(4, 5), (5, 1)],
                             [(4, 7), (3, 1)], [(4, 3), (0, 7)]]

    def __str__(self) -> str:
        return str(self.state)

    def U(self) -> None:
        """
        Rotate face 1 clockwise.
        """
        trans_12 = [(0, [0, 1, 2]), (3, [8, 7, 6]), (5, [8, 7, 6]), (2, [0, 1, 2])]
        trans_8 = [(1, [0, 1]), (1, [2, 5]), (1, [8, 7]), (1, [6, 3])]
        self.transition(trans_12)
        self.transition(trans_8)

    def D(self) -> None:
        """
        Rotate face 4 clockwise.
        """
        trans_12 = [(0, [6, 7, 8]), (2, [6, 7, 8]), (5, [2, 1, 0]), (3, [2, 1, 0])]
        trans_8 = [(4, [0, 1]), (4, [2, 5]), (4, [8, 7]), (4, [6, 3])]
        self.transition(trans_12)
        self.transition(trans_8)

    def F(self) -> None:
        """
        Rotate face 0 clockwise.
        """
        trans_12 = [(1, [6, 7, 8]), (2, [0, 3, 6]), (4, [0, 3, 6]), (3, [0, 3, 6])]
        trans_8 = [(0, [0, 1]), (0, [2, 5]), (0, [8, 7]), (0, [6, 3])]
        self.transition(trans_12)
        self.transition(trans_8)

    def B(self) -> None:
        """
        Rotate face 5 clockwise.
        """
        trans_12 = [(1, [0, 1, 2]), (3, [2, 5, 8]), (4, [2, 5, 8]), (2, [2, 5, 8])]
        trans_8 = [(5, [0, 1]), (5, [2, 5]), (5, [8, 7]), (5, [6, 3])]
        self.transition(trans_12)
        self.transition(trans_8)

    def L(self) -> None:
        """
        Rotate face 3 clockwise.
        """
        trans_12 = [(0, [0, 3, 6]), (4, [6, 7, 8]), (5, [0, 3, 6]), (1, [0, 3, 6])]
        trans_8 = [(3, [0, 1]), (3, [2, 5]), (3, [8, 7]), (3, [6, 3])]
        self.transition(trans_12)
        self.transition(trans_8)

    def R(self) -> None:
        """
        Rotate face 2 clockwise.
        """
        trans_12 = [(0, [2, 5, 8]), (1, [2, 5, 8]), (5, [2, 5, 8]), (4, [0, 1, 2])]
        trans_8 = [(2, [0, 1]), (2, [2, 5]), (2, [8, 7]), (2, [6, 3])]
        self.transition(trans_12)
        self.transition(trans_8)

    def M(self) -> None:
        """
        Rotate the middle layer clockwise (from the left).
        """
        trans_12 = [(0, [1, 4, 7]), (4, [3, 4, 5]), (5, [1, 4, 7]), (1, [1, 4, 7])]
        self.transition(trans_12)
    
    def E(self) -> None:
        """
        Rotate the equatorial layer clockwise (from the bottom).
        """
        trans_12 = [(0, [3, 4, 5]), (2, [3, 4, 5]), (5, [5, 4, 3]), (3, [5, 4, 3])]
        self.transition(trans_12)

    def S(self) -> None:
        """
        Rotate the standing layer clockwise (from the front).
        """
        trans_12 = [(1, [3, 4, 5]), (2, [1, 4, 7]), (4, [1, 4, 7]), (3, [1, 4, 7])]
        self.transition(trans_12)
        
    def transition(self, trans: list[tuple]) -> None:
        """
        Modify the given state with the given transition list.
        If trans = [(face_1, [facelet_1a, facelet_1b, ... ]), ... ],
        then move state[face_n, facelet_na] to state[face_1, facelet_1a], move state[face_1, facelet_1a] to state[face_2, facelet_2a], ...
        and move state[face_n, facelet_nb] to state[face_1, facelet_1b], move state[face_1, facelet_1b] to state[face_2, facelet_2b], ... an so on.
        Args:
            trans: facelets to move in each face.
        """
        for state in [self.state, self.index_state]:
            prev_state = np.copy(state)
            for i in range(len(trans)):
                face_1, face_2 = trans[i-1], trans[i]
                for facelet_1a, facelet_2a in zip(face_1[1], face_2[1]):
                    state[face_2[0], facelet_2a] = prev_state[face_1[0], facelet_1a]

    def move(self, sequence: Iterable[str]) -> None:
        """
        Move with the given sequence of moves.
        """
        for move in sequence:
            if len(move) == 1:
                repeat = 1
            elif len(move) == 2 and move[1] == "2":
                repeat = 2
            elif len(move) == 2 and move[1] == "'":
                repeat = 3
            else:
                raise ValueError("undefined move: %s" % move)

            for _ in range(repeat):
                if move[0] == "U":
                    self.U()
                elif move[0] == "D":
                    self.D()
                elif move[0] == "F":
                    self.F()
                elif move[0] == "B":
                    self.B()
                elif move[0] == "L":
                    self.L()
                elif move[0] == "R":
                    self.R()
                elif move[0] == "M":
                    self.M()
                elif move[0] == "E":
                    self.E()
                elif move[0] == "S":
                    self.S()
                else:
                    raise ValueError("undefined move: %s" % move)

    def shuffle(self, num: Optional[int] = 60) -> None:
        """
        Randomly shuffle the cube with the given number of shuffling.
        Note that this function only gives the solvable states of the cube.
        """
        move_clock = ['U', 'D', 'F', 'B', 'L', 'R', 'M', 'E', 'S']
        move_double = ['U2', 'D2', 'F2', 'B2', 'L2', 'R2', 'M2', 'E2', 'S2']
        move_counter_clock = ["U'", "D'", "F'", "B'", "L'", "R'", "M'", "E'", "S'"]
        moves = move_clock + move_double + move_counter_clock
        sequence = np.random.choice(moves, num)
        self.move(sequence)

    def reset(self) -> None:
        """
        Reset the state of the cube to the initial (solved) state.
        """
        self.__init__()

    def mask_corner(self, num_mask = 1) -> None:
        """
        Randomly mask corner facelets.
        """

        one_masked_corners = {} # Corner number (0~7) -> Facelet number (0~2)
        two_masked_corners = [] # Corner number
        corner_colors = self.corners()

        while num_mask > 0:
            corner = np.random.randint(8)
            if corner in two_masked_corners:
                continue
            elif corner in one_masked_corners:
                unmasked_facelet = np.random.permutation(3).tolist()
                unmasked_facelet.remove(one_masked_corners[corner])
                facelet_1, facelet_2 = unmasked_facelet
                two_masked_corner_colors = corner_colors[two_masked_corners]
                if corner_colors[corner][facelet_1] not in two_masked_corner_colors:
                    self.state[self.corner_indices[corner][facelet_2]] = -1
                    one_masked_corners.pop(corner)
                    two_masked_corners.append(corner)
                elif corner_colors[corner][facelet_2] not in two_masked_corner_colors:
                    self.state[self.corner_indices[corner][facelet_1]] = -1
                    one_masked_corners.pop(corner)
                    two_masked_corners.append(corner)
                num_mask += -1
            else:
                facelet = np.random.randint(3)
                self.state[self.corner_indices[corner][facelet]] = -1
                one_masked_corners[corner] = facelet
                num_mask += -1

    def mask_edge(self, num_mask: Optional[int] = 1) -> None:
        """
        Randomly mask edge facelets.
        """
        while num_mask > 0:
            face = np.random.randint(6)
            facelet = np.random.choice([1, 3, 5, 7])
            if self.state[face, facelet] >= 0:
                self.state[face, facelet] = -1
                num_mask += -1

    def mask_center(self, num_mask: Optional[int] = 1) -> None:
        """
        Randomly mask center facelets.
        """
        while num_mask > 0:
            face = np.random.randint(6)
            facelet = 4
            if self.state[face, facelet] >= 0:
                self.state[face, facelet] = -1
                num_mask += -1

    def to_sat(self) -> np.ndarray:
        """
        Generate the sat assignment corresponding to the state.
        """
        sat = np.zeros((6, 9, 6))
        for i in range(6):
            for j in range(9):
                if self.state[i, j] >= 0:
                    sat[i, j, self.state[i, j]] = 1
        return sat

    def from_sat(self, assign: np.ndarray) -> None:
        """
        Set the state by the given sat assignment.
        The input assignment may contain continuous probabilities.
        """
        self.state = np.argmax(assign, 2)

    def corners(self) -> np.ndarray:
        """
        Return the colors of the corner cubies.
        """
        corner_colors = [[self.state[index] for index in triple] for triple in self.corner_indices]
        return np.array(corner_colors)

    def edges(self) -> np.ndarray:
        """
        Return the colors of the edge cubies.
        """
        edges = [[self.state[index] for index in pair] for pair in self.edge_indices]
        return np.array(edges)

    def centers(self) -> np.ndarray:
        """
        Return the colors of the center cubies.
        """
        centers = [self.state[face, 4] for face in range(6)]
        return np.array(centers)


def main():
    num_generate = 10000
    num_moves = 60
    num_mask_corner = 3
    num_mask_edge = 1
    num_mask_center = 1

    cube = Cube333()
    features, labels = [], []

    for _ in range(num_generate):
        # Generate labels
        cube.shuffle(num_moves)
        labels.append(cube.to_sat())

        # Generate features (by masking some facelets)
        cube.mask_corner(num_mask_corner)
        cube.mask_edge(num_mask_edge)
        cube.mask_center(num_mask_center)
        features.append(cube.to_sat())
        cube.reset()

    features, labels = torch.Tensor(features), torch.Tensor(labels)

    save_dir = "cube"
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)

    torch.save(features, os.path.join(save_dir, 'features.pt'))
    torch.save(labels, os.path.join(save_dir, 'labels.pt'))


if __name__ == "__main__":
    main()
