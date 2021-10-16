"""
eight_puzzle.py

Contains a class for representing an 8-puzzle
and classes defining the puzzle solver.

"""

import itertools
import collections

import numpy as np

from modules.puzzles.config import solve_return


class Node:
    """
    A class representing an Solver node
    - 'puzzle' is a Puzzle instance
    - 'parent' is the preceding node generated by the solver, if any
    - 'action' is the action taken to produce puzzle, if any
    """

    def __init__(self, puzzle, parent=None, action=None):
        self.puzzle = puzzle
        self.parent = parent
        self.action = action
        if self.parent is not None:
            self.g = parent.g + 1
        else:
            self.g = 0

    @property
    def score(self):
        return self.g + self.h

    @property
    def state(self):
        """
        Return a hashable representation of self
        """
        return str(self)

    @property
    def path(self):
        """
        Reconstruct a path from to the root 'parent'
        """
        node, p = self, []
        while node:
            p.append(node)
            node = node.parent
        return p

    @property
    def solved(self):
        """Wrapper to check if 'puzzle' is solved"""
        return self.puzzle.solved

    @property
    def actions(self):
        """Wrapper for 'actions' accessible at current state"""
        return self.puzzle.actions

    @property
    def h(self):
        """ "h"""
        return self.puzzle.manhattan

    @property
    def f(self):
        """ "f"""
        return self.h + self.g

    def __str__(self):
        return str(self.puzzle)


class Solver:
    """
    An '8-puzzle' solver
    - 'start' is a Puzzle instance
    """

    def __init__(self, initial_board):
        self.initial_board = initial_board

    def solve(self, Q):
        """
        Perform A* search and return a path
        to the solution, if it exists
        """
        solve_return = Q.get()
        queue = collections.deque([Node(self.initial_board)])
        seen = set()
        seen.add(queue[0].state)
        while queue:
            queue = collections.deque(sorted(list(queue), key=lambda node: node.f))
            node = queue.popleft()
            if node.solved:
                solve_return = node.path
                Q.put(solve_return)
                break

            for move, action in node.actions:
                child = Node(move(), node, action)

                if child.state not in seen:
                    queue.appendleft(child)
                    seen.add(child.state)


class Puzzle:
    """
    A class representing an '8-puzzle'.
    - 'board' should be a square list of lists
    with integer entries 0...width^2 - 1
    e.g. [[1,2,3],[4,0,6],[7,5,8]]
    """

    def __init__(self, board):
        self.width = board.shape[0]
        self.board = board

    @property
    def valid(self):
        """
        Check if a puzzle contains all digits from 0 to 8,
        digit 9 is not detected, and each digit appears once
        only
        """
        unique = len(np.unique(self.board)) == len(self.board.flatten())
        nine_detected = 9 in self.board

        return unique and not nine_detected

    @property
    def solvable(self):
        """
        Check if puzzle is solvable
        """
        # Count inversions in given 8 puzzle
        inv_count = self._get_inv_count(self.board.flatten())

        # return true if inversion count is even.
        return inv_count % 2 == 0

    @property
    def solved(self):
        """
        The puzzle is solved if the flattened board's numbers are in
        increasing order from left to right and the '0' tile is in the
        last position on the board
        """
        N = self.width * self.width
        return str(self) == "".join(map(str, range(1, N))) + "0"

    @property
    def actions(self):
        """
        Return a list of 'move', 'action' pairs. 'move' can be called
        to return a new puzzle that results in sliding the '0' tile in
        the direction of 'action'.
        """

        def create_move(at, to):
            return lambda: self._move(at, to)

        moves = []
        for i, j in itertools.product(range(self.width), range(self.width)):
            direcs = {
                "RIGHT": (i, j - 1),
                "LEFT": (i, j + 1),
                "DOWN": (i - 1, j),
                "UP": (i + 1, j),
            }

            for action, (r, c) in direcs.items():
                if (
                    r >= 0
                    and c >= 0
                    and r < self.width
                    and c < self.width
                    and self.board[r, c] == 0
                ):
                    move = create_move((i, j), (r, c)), action
                    moves.append(move)
        return moves

    @property
    def manhattan(self):
        distance = 0
        for i in range(3):
            for j in range(3):
                if self.board[i, j] != 0:
                    x, y = divmod(self.board[i, j] - 1, 3)
                    distance += abs(x - i) + abs(y - j)
        return distance

    def copy(self):
        """
        Return a new puzzle with the same board as 'self'
        """
        board = np.copy(self.board)
        return Puzzle(board)

    def _move(self, at, to):
        """
        Return a new puzzle where 'at' and 'to' tiles have been swapped.
        NOTE: all moves should be 'actions' that have been executed
        """
        copy = self.copy()
        i, j = at
        r, c = to
        copy.board[i, j], copy.board[r, c] = copy.board[r, c], copy.board[i, j]
        return copy

    def _get_inv_count(self, board_flat):
        """Count inversions in a given puzzle board"""
        inv_count = 0
        empty_value = 0
        for i in range(0, 9):
            for j in range(i + 1, 9):
                if (
                    board_flat[j] != empty_value
                    and board_flat[i] != empty_value
                    and board_flat[i] > board_flat[j]
                ):
                    inv_count += 1
        return inv_count

    def __str__(self):
        return "".join(map(str, self))

    def __iter__(self):
        for row in self.board:
            yield from row
