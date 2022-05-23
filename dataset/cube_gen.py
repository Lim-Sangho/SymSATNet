# %%
from __future__ import annotations
from typing import Iterable, Optional
import matplotlib.pyplot as plt
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

        # self.corner_state = np.arange(8)
        # self.orientation_state = np.array([[1, 1, 2, 1, 0, 1, 2, 1, 1],
        #                                    [0, 0, 0, 0, 0, 0, 0, 0, 0],
        #                                    [1, 1, 2, 0, 0, 0, 2, 1, 1],
        #                                    [1, 1, 2, 0, 0, 0, 2, 1, 1],
        #                                    [0, 0, 0, 0, 0, 0, 0, 0, 0],
        #                                    [1, 1, 2, 1, 0, 1, 2, 1, 1]])

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

        # self.cubie_state = np.copy(self.state)
        # for i, corner in enumerate(self.corner_indices):
        #     for facelet in corner:
        #         self.cubie_state[facelet] = i
        # for j, edge in enumerate(self.edge_indices):
        #     for facelet in edge:
        #         self.cubie_state[facelet] = j

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

        # trans_corner_number = [0, 1, 2, 3]
        # self.transition_corner_number(trans_corner_number)

    def D(self) -> None:
        """
        Rotate face 4 clockwise.
        """
        trans_12 = [(0, [6, 7, 8]), (2, [6, 7, 8]), (5, [2, 1, 0]), (3, [2, 1, 0])]
        trans_8 = [(4, [0, 1]), (4, [2, 5]), (4, [8, 7]), (4, [6, 3])]
        self.transition(trans_12)
        self.transition(trans_8)

        # trans_corner_number = [4, 5, 6, 7]
        # self.transition_corner_number(trans_corner_number)


    def F(self) -> None:
        """
        Rotate face 0 clockwise.
        """
        trans_12 = [(1, [6, 7, 8]), (2, [0, 3, 6]), (4, [0, 3, 6]), (3, [0, 3, 6])]
        trans_8 = [(0, [0, 1]), (0, [2, 5]), (0, [8, 7]), (0, [6, 3])]
        self.transition(trans_12)
        self.transition(trans_8)

        # trans_corner_number = [3, 2, 4, 7]
        # self.transition_corner_number(trans_corner_number)

    def B(self) -> None:
        """
        Rotate face 5 clockwise.
        """
        trans_12 = [(1, [0, 1, 2]), (3, [2, 5, 8]), (4, [2, 5, 8]), (2, [2, 5, 8])]
        trans_8 = [(5, [0, 1]), (5, [2, 5]), (5, [8, 7]), (5, [6, 3])]
        self.transition(trans_12)
        self.transition(trans_8)

        # trans_corner_number = [6, 5, 1, 0]
        # self.transition_corner_number(trans_corner_number)

    def L(self) -> None:
        """
        Rotate face 3 clockwise.
        """
        trans_12 = [(0, [0, 3, 6]), (4, [6, 7, 8]), (5, [0, 3, 6]), (1, [0, 3, 6])]
        trans_8 = [(3, [0, 1]), (3, [2, 5]), (3, [8, 7]), (3, [6, 3])]
        self.transition(trans_12)
        self.transition(trans_8)

        # trans_corner_number = [7, 6, 0, 3]
        # self.transition_corner_number(trans_corner_number)

    def R(self) -> None:
        """
        Rotate face 2 clockwise.
        """
        trans_12 = [(0, [2, 5, 8]), (1, [2, 5, 8]), (5, [2, 5, 8]), (4, [0, 1, 2])]
        trans_8 = [(2, [0, 1]), (2, [2, 5]), (2, [8, 7]), (2, [6, 3])]
        self.transition(trans_12)
        self.transition(trans_8)

        # trans_corner_number = [2, 1, 5, 4]
        # self.transition_corner_number(trans_corner_number)

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

    # def transition_corner_number(self, trans: list) -> None:
    #     """
    #     Modify the corner number state with the given transition list.
    #     """
    #     prev_corner_state = np.copy(self.corner_state)
    #     for i in range(len(trans)):
    #         self.corner_state[trans[i]] = prev_corner_state[trans[i-1]]

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

    def twist(self, num: Optional[int] = 60) -> None:
        """
        Randomly twist the orientations and permute the cubies.
        Note that this function may give unsolvable states of the cube, but without invalid cubies,
        i.e., it gives a random state among the 12 orbits of the Rubik's cube.
        """
        trans_corner = [(0, [2]), (1, [8]), (2, [0])]
        trans_edge = [(0, [1]), (1, [7])]
        perm_1 = [(1, [7]), (1, [5])]
        perm_2 = [(0, [1]), (2, [1])]

        for _ in range(np.random.randint(0, 3)):
            self.transition(trans_corner)
        if np.random.randint(0, 2):
            self.transition(trans_edge)
        if np.random.randint(0, 2):
            self.transition(perm_1)
            self.transition(perm_2)

        self.shuffle(num)

    def reset(self) -> None:
        """
        Reset the state of the cube to the initial (solved) state.
        """
        self.__init__()

    def mask_corner(self, num_mask = 1) -> None:
        """
        Randomly mask corner facelets.
        """
        # while num_mask > 0:
        #     face = np.random.randint(6)
        #     facelet = np.random.choice([0, 2, 6, 8])
        #     if self.state[face, facelet] >= 0:
        #         self.state[face, facelet] = -1
        #         num_mask += -1

        one_masked_corners = {} # Corner number (0~7) -> Facelet number (0~2)
        two_masked_corners = [] # Corner number
        corner_colors = self.corners()

        while num_mask > 0:
            corner = np.random.randint(8)
            if corner in two_masked_corners:
                continue
            elif corner in one_masked_corners:
                # continue
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

    # def mask_corner_cubie(self, num_mask: Optional[int] = 1) -> None:
    #     """
    #     Randomly mask corner cubie numbers.
    #     """
    #     while num_mask > 0:
    #         corner = np.random.randint(8)
    #         facelet = np.random.randint(3)
    #         index = self.corner_indices[corner][facelet]
    #         if self.cubie_state[index] >= 0:
    #             self.cubie_state[index] = -1
    #             num_mask += -1

    # def mask_edge_cubie(self, num_mask: Optional[int] = 1) -> None:
    #     """
    #     Randomly mask edge cubie numbers.
    #     """
    #     while num_mask > 0:
    #         edge = np.random.randint(12)
    #         facelet = np.random.randint(2)
    #         index = self.edge_indices[edge][facelet]
    #         if self.cubie_state[index] >= 0:
    #             self.cubie_state[index] = -1
    #             num_mask += -1

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

    def to_sat_global(self) -> np.ndarray:
        """
        Generate the sat assignment corresponding to the state and cubie_state.
        """
        sat = np.zeros((6, 9, 26))
        for i in range(6):
            for j in range(9):
                if self.state[i, j] >= 0:
                    sat[i, j, self.state[i, j]] = 1
        
        for corner in self.corner_indices:
            for facelet in corner:
                cubie = self.cubie_state[facelet]
                if cubie >= 0:
                    sat[facelet + (6 + cubie,)] = 1

        for edge in self.edge_indices:
            for facelet in edge:
                cubie = self.cubie_state[facelet]
                if cubie >= 0:
                    sat[facelet + (14 + cubie,)] = 1

        return sat
        

    # def to_sat_corner_number(self) -> np.ndarray:
    #     """
    #     Generate the truth variables of the unique number of each corner cubie.
    #     This requires 8 * 8 = 64 more truth variables.
    #     """
    #     sat = np.zeros((8, 8))
    #     for i in range(8):
    #         sat[i, self.corner_state[i]] = 1
    #     return sat

    # def to_sat_orientation(self) -> np.ndarray:
    #     """
    #     Generate the sat assignment corresponding to the orientation state.
    #     """
    #     sat = np.zeros((6, 9, 3))
    #     for i in range(6):
    #         for j in range(9):
    #             sat[i, j, self.orientation_state[i, j]] = 1
    #     return sat

    # def from_sat_corner_number(self, assign: np.ndarray) -> None:
    #     """
    #     Set the state by the given sat assignment with global information.
    #     The input assignment may contain continuous probabilities.
    #     """
    #     self.corner_state = np.argmax(assign, 1)

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

    def perm_corner(self, corners: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Return the corner permutation of the cube.
        We use the characteristic numbers of the corner cubies each of which is the sum of the 3 colors of each cubie.
        """
        init = [9, 8, 3, 4, 6, 11, 12, 7] # Characteristic numbers (sum of 3 colors) of corners of a solved Rubik's cube.
        corners = self.corners() if not isinstance(corners, np.ndarray) else corners
        corners = np.sum(corners, axis = 1)
        return np.array([corners.tolist().index(i) for i in init])

    def perm_edge(self, edges: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Return the edge permutation of the cube.
        We use the characteristic numbers of the edge cubies each of which is the sum of squares of the 2 colors of each cubie.
        """
        init = [26, 5, 1, 10, 4, 29, 9, 34, 20, 41, 25, 16] # Characteristic numbers (sum of squares of 2 colors) of edges of a solved Rubik's cube.
        edges = self.edges() if not isinstance(edges, np.ndarray) else edges
        edges = np.sum(np.square(edges), axis = 1)
        return np.array([edges.tolist().index(i) for i in init])

    def perm_center(self, centers: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Return the center permutation of the cube.
        """
        centers = self.centers() if not isinstance(centers, np.ndarray) else centers
        return np.array([centers.tolist().index(i) for i in range(6)])

    def perm_sign(self, perm: np.ndarray) -> int:
        """
        Return the sign of the given permutation.
        """
        perm = perm.tolist()
        indices = list(range(len(perm)))
        cycles = []
        
        while len(indices) > 0:
            cycle = [indices.pop(0)]
            while perm.index(cycle[-1]) in indices:
                indices.remove(perm.index(cycle[-1]))
                cycle.append(perm.index(cycle[-1]))
            cycles.append(cycle)

        return (len(perm) - len(cycles)) % 2
    
    def orient_corner(self, corners: Optional[np.ndarray] = None, perm: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Return the corner orientation of the cube.
        """
        init = [[1, 3, 5], [1, 5, 2], [1, 2, 0], [1, 0, 3], [4, 0, 2], [4, 2, 5], [4, 5, 3], [4, 3, 0]]
        corners = self.corners() if not isinstance(corners, np.ndarray) else corners
        perm = self.perm_corner(corners) if not isinstance(perm, np.ndarray) else perm
        orientations = [init[i].index(corners[j][0]) for i, j in enumerate(perm)]
        return orientations

    def orient_edge(self, edges: Optional[np.ndarray] = None, perm: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Return the edge orientation of the cube.
        """
        init = [[1, 5], [1, 2], [1, 0], [1, 3], [2, 0], [2, 5], [3, 0], [3, 5], [4, 2], [4, 5], [4, 3], [4, 0]]
        edges = self.edges() if not isinstance(edges, np.ndarray) else edges
        perm = self.perm_edge(edges) if not isinstance(perm, np.ndarray) else perm
        orientations = [init[i].index(edges[j][0]) for i, j in enumerate(perm)]
        return orientations

    def check_corner(self, corners: Optional[np.ndarray] = None) -> bool:
        """
        Check whether the corner cubies are valid.
        """
        init = [-12, -12, -6, -6, 2, 2, 16, 16]
        corners = self.corners() if not isinstance(corners, np.ndarray) else corners
        corners = [(corner[0] - corner[1]) * (corner[1] - corner[2]) * (corner[2] - corner[0]) for corner in corners]
        return sorted(corners) == init

    def check_edge(self, edges: Optional[np.ndarray] = None) -> bool:
        """
        Check whether the edge cubies are valid.
        """
        init = {26, 5, 1, 10, 4, 29, 9, 34, 20, 41, 25, 16}
        edges = self.edges() if not isinstance(edges, np.ndarray) else edges
        edges = np.sum(np.square(edges), axis = 1)
        return set(edges) == init

    def check_center(self, centers: Optional[np.ndarray] = None) -> bool:
        """
        Check whether the center cubies are valid.
        """
        centers = self.centers() if not isinstance(centers, np.ndarray) else centers
        return set(centers) == set(range(6))
        
    def is_valid(self) -> bool:
        """
        Check whether the state of the cube is vaild.
        """
        corners = self.corners()
        edges = self.edges()
        centers = self.centers()

        # if not (self.check_corner(corners) and self.check_edge(edges) and self.check_center(centers)):
        #     # Some cubies are invalid.
        #     return False

        try:
            perm_corner = self.perm_corner(corners)
            perm_edge = self.perm_edge(edges)
            perm_center = self.perm_center(centers)
            orient_corner = self.orient_corner(corners, perm_corner)
            orient_edge = self.orient_edge(edges, perm_edge)

            is_sign_valid = (self.perm_sign(perm_corner) + self.perm_sign(perm_edge) + self.perm_sign(perm_center)) % 2 == 0
            is_orient_valid = sum(orient_corner) % 3 == 0 and sum(orient_edge) % 2 == 0
            is_center_valid = ((centers[1] - centers[0]) * (centers[0] - centers[2]) * (centers[2] - centers[1]) in [-16, -2, 6, 12] and
                            (centers[0] + centers[5]) == (centers[1] + centers[4]) == (centers[2] + centers[3]))

            return is_sign_valid and is_orient_valid and is_center_valid

        except ValueError:
            # Some cubies are invalid.
            return False


def main():
    num_generate = 10000
    num_moves = 60
    num_mask_corner = 3
    num_mask_edge = 1
    num_mask_center = 1

    cube = Cube333()
    features, labels = [], []

    for _ in range(num_generate):
        cube.shuffle(num_moves)
        assert(cube.is_valid())

        # cube.twist(num_moves)
        # assert(cube.check_corner() and cube.check_edge() and cube.check_center())

        labels.append(cube.to_sat())

        cube.mask_corner(num_mask_corner)
        cube.mask_edge(num_mask_edge)
        cube.mask_center(num_mask_center)
        features.append(cube.to_sat())
        cube.reset()

    features, labels = map(torch.Tensor, [features, labels])

    save_dir = "cube_toy"
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)

    torch.save(features, os.path.join(save_dir, 'features.pt'))
    torch.save(labels, os.path.join(save_dir, 'labels.pt'))


def draw(D, color = 'jet', bar = True, dpi = 400):
    plt.figure(dpi=dpi)
    plt.imshow(D, cmap=plt.get_cmap(color))
    if bar: plt.colorbar()
    plt.show()


if __name__ == "__main__":
    main()

#     cube = Cube333()
#     string = '''[[2 3 4 2 3 3 0 1 0]
#  [0 1 3 3 0 4 5 0 2]
#  [5 5 4 1 4 0 1 3 3]
#  [1 4 5 5 1 1 1 4 3]
#  [2 5 1 0 5 0 2 2 3]
#  [4 2 5 5 2 4 4 2 0]]'''
#     state = np.fromstring(" ".join(string.replace("[", "").replace("]", "").split()), dtype = int, sep = " ").reshape(6, 9)
#     cube.state = state
#     print(cube.corners())
# %%
