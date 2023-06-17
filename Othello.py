from argparse import ArgumentParser
import re

import numpy as np

class othelloError(ValueError): pass
class IllegalMove(othelloError): pass
class InvalidMove(othelloError): pass

class Space:
    def __init__(self, i, j):
        self.i, self.j = i, j
        self.neighbors = {}
        self.player = ''

    def find_neighbors(self, grid):
        N = len(grid)
        neighbors = self.neighbors
        for di in (-1, 0, 1):
            for dj in (-1, 0, 1):
                I, J = self.i + di, self.j + dj
                if not (0 <= I < N and 0 <= J < N):
                    neighbors[di,dj] = None
                else:
                    neighbors[di,dj] = grid[I,J]

    def set(self, player, force=False):
        flips = self.checkmove(player, force=force)
        self.player = player
        for space in flips:
            space.player = player

    def checkmove(self, player, force=False):
        if self.player and not force:
            raise IllegalMove(f'space ({self.i}, {self.j}) already occupied by {self.player}')
        dirs = [(-1,0), (1,0), (0,-1), (0,1)]
        goods = []
        for di in (-1, 0, 1):
            for dj in (-1, 0, 1):
                n = self.next(di,dj)
                if n and n.player and n.player != player:
                    goods.append(n)
        if not goods and not force:
            raise IllegalMove(f'illegal move: must place adjacent to opponent')
        flips = []
        for di in (-1, 0, 1):
            for dj in (-1, 0, 1):
                cur = self
                chain = [cur]
                while cur.next(di,dj):
                    cur = cur.next(di,dj)
                    if not cur.player:
                        break
                    chain.append(cur)
                    if cur.player == player:
                        break
                if chain[-1].player == player:
                    for space in chain[1:-1]:
                        flips.append(space)
        if not flips and not force:
            raise IllegalMove(f'illegal move: must sandwich opponent')
        return flips

    def next(self, di, dj):
        return self.neighbors[di,dj]


class Board:

    def __init__(self, N=8, othello=True):
        self.othello = othello
        self.N = N
        self.grid = np.array([
            [Space(i, j) for j in range(N)]
            for i in range(N)
        ])
        for space in np.ravel(self.grid):
            space.find_neighbors(self.grid)
        if othello:
            self('B', 3, 4, force=True)
            self('W', 3, 3, force=True)
            self('B', 4, 3, force=True)
            self('W', 4, 4, force=True)

    def __getitem__(self, *x):
        return self.grid.__getitem__(*x)

    def playergrid(self):
        grid, N = self.grid, self.N
        return np.array([
            [grid[i,j].player for j in range(N)]
            for i in range(N) ])

    def __str__(self):
        grid, N = self.grid, self.N
        lines = [''.join('({:1})'.format(grid[i,j].player) for j in range(N))
                  for i in range(N)]
        lines += ['  ' + ''.join(' {} '.format(i+1) for i in range(N))]
        for (i,line) in enumerate(lines[:-1]):
            lines[i] = '{} '.format(chr(ord('a') + i)) + line
        return '\n'.join(lines)

    def __call__(self, player, i, j, force=False):
        self[i,j].set(player, force=force)


class Game:

    def __init__(self, N=8, othello=True,
                 skill_W=0, skill_B=0,
                 best_W=False, best_B=False,
                 rand_W=False, rand_B=False):
        self.board = Board(N=N, othello=othello)
        self.cur_player = 'B'
        self.auto = ''
        if skill_W or best_W or rand_W:
            self.auto += 'W'
        if skill_B or best_B or rand_B:
            self.auto += 'B'
        self.skill = dict(W=skill_W, B=skill_B)
        self.best = dict(W=best_W, B=best_B)
        self.rand = dict(W=rand_W, B=rand_B)
        print(self.auto)

    def move(self, s):
        try:
            rows = re.findall('[a-h]', s)
            assert len(rows) == 1
            i = ord(rows[0]) - ord('a')
            cols = re.findall('[1-8]', s)
            assert len(cols) == 1
            j = int(cols[0]) - 1
        except Exception as e:
            raise InvalidMove(f'could not parse move: "{s}"') from e
        player = self.cur_player
        self.board(player, i, j)
        self.toggleplayer()

    def toggleplayer(self):
        self.cur_player = 'W' if self.cur_player == 'B' else 'B'
        return self.cur_player

    def score(self):
        pgrid = self.board.playergrid()
        B = np.sum(pgrid == 'B')
        W = np.sum(pgrid == 'W')
        return B, W


    def __str__(self):
        B, W = self.score()
        hB = 'B: {}'.format(B)
        hW = 'W: {}'.format(W)
        lB = len(hB)
        lW = len(hW)
        width = 2 + 3 * self.board.N
        header = hB + (width - lB - lW) * ' ' + hW
        div = width * '-'
        out = '\n'.join([div, header, div, str(self.board).lower()])
        return out

    def get_legal_moves(self, player):
        board = self.board
        N = board.N
        moves = []
        n_flips = []
        for i in range(N):
            for j in range(N):
                try:
                    f = board[i,j].checkmove(player)
                    moves.append((i,j))
                    n_flips.append(len(f))
                except IllegalMove:
                    pass
        return moves, np.array(n_flips)

    def play(self):
        orig_player = player = self.cur_player
        print(self)
        hasmoves = self.get_legal_moves(player)[0]
        if not hasmoves:
            player = self.toggleplayer()
            hasmoves = self.get_legal_moves(player)[0]
            if hasmoves:
                print(f'Player "{orig_player}" has no moves; passing to Player "{player}"')
            else:
                print(f'No legal moves remain!')
                self.endgame()
                return

        try:
            if player in self.auto:
                m = self.autoplay()
            else:
                m = input(f'{player} => ')
        except WBError:
            print('\nGame Ended.')
            self.endgame()
            return
        if m in 'quit'[:len(m)] or m == '':
            print('\nGame Ended.')
            self.endgame()
            return
        print()
        try:
            self.move(m)
        except othelloError as e:
            print(e)
            pass
        self.play()

    def autoplay(self):
        player = self.cur_player
        moves, n_flips = self.get_legal_moves(player)
        if self.skill[player]:
            p = n_flips ** self.skill[player]
            p /= p.sum()
            i, j = moves[np.random.choice(len(moves), p=p)]
        if self.best[player]:
            i, j = moves[np.argmax(n_flips)]
        else:
            i, j = moves[np.random.choice(len(moves))]
        move = f'{i+1} {j+1}'
        move = f'{chr(ord("a")+i)} {j+1}'
        print(f'{player} => {move} (*computer player*)')
        return move



    def endgame(self):
        B, W = self.score()
        if B == W:
            print(f'Final score is a tie: {B} - {W}.')
        elif B > W:
            print(f'Player "B" wins by {B} - {W}.')
        else:
            print(f'Player "W" wins by {W} - {B}.')


if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument('--skill-B', default=0, type=float)
    parser.add_argument('--best-B', default=False, action='store_true')
    parser.add_argument('--rand-B', default=False, action='store_true')
    parser.add_argument('--skill-W', default=0, type=float)
    parser.add_argument('--best-W', default=False, action='store_true')
    parser.add_argument('--rand-W', default=False, action='store_true')
    opts = parser.parse_args()

    auto = dict(skill_W=opts.skill_W, skill_B=opts.skill_B,
                best_W=opts.best_W, best_B=opts.best_B,
                rand_W=opts.rand_W, rand_B=opts.rand_B)
    g = Game(**auto)
    g.play()
