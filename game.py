import numpy as np
import gym
from gym import spaces
#from gym.utils import seeding
#from gym.envs.classic_control import rendering
import os
from subprocess import Popen, PIPE

LEFT = 0b00
RIGHT = 0b01
UP = 0b10
DOWN = 0b11
action_str = {
        LEFT : "LEFT",
        RIGHT : "RIGHT",
        UP : "UP",
        DOWN : "DOWN",
        }

VERT_BIT = 0x10
POS_BIT = 0x01

class Game(gym.Env):
    metadata = {
            'render.modes' : ['human']
            }
    def __init__(self, w, h):
        self.w, self.h = w, h
        self.board = np.zeros((h,w), dtype=np.uint8)

        ir = range(self.h)
        jr = range(self.w)
        self.range = {
                LEFT : (ir,jr),
                RIGHT : (ir, list(reversed(jr))),
                UP : (ir,jr),
                DOWN : (list(reversed(ir)), jr),
                }


        low = np.zeros((self.w, self.h), dtype=np.uint8)
        high = np.full((self.w, self.h), 16, dtype=np.uint8)

        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(low,high)

        self.reset()

        self.disp_path = '/tmp/2048.log'
        self.viewer = None

    def spawn(self, m=1):
        i_idx,j_idx = np.where(self.board == 0)
        n = len(i_idx)
        s = np.random.choice(n, size=m, replace=False)
        v = 1 + np.random.choice(2, size=m, p=(0.75, 0.25), replace=True)
        self.board[i_idx[s], j_idx[s]] = v
    def state(self, copy=True):
        return self.board.copy()
        #res = [(self.board == i).astype(np.float32) for i in range(16)]
        #return np.dstack(res)
        #return np.array(res, dtype=np.float32)
        #res = self.board.flatten() / 16.0
        #if copy:
        #    return res.copy()
        #else:
        #    return res

    def _reset(self):
        self.board.fill(0)
        self.spawn(2)
        return self.state()

    def _step(self, action):
        ir,jr = self.range[action]
        board = self.board.T if (action & VERT_BIT) else self.board
        ir,jr = (jr,ir) if (action & VERT_BIT) else (ir,jr)
        spawn_flag = False

        for i in ir:
            ref_idx = (i, jr[0])
            ref_val = board[ref_idx]
            for j in jr[1:]:
                val = board[i][j]
                if val > 0:

                    ref_j = ref_idx[1]
                    nxt_j = (ref_j-1 if (action & POS_BIT) else ref_j+1)
                    nxt_idx = (i, nxt_j)

                    if ref_val == 0: # replace
                        spawn_flag = True
                        board[i][j] = 0 
                        board[ref_idx] = val
                        ref_val = val
                    elif ref_val == val: # merge
                        spawn_flag = True
                        board[i][j] = 0
                        board[ref_idx] += 1
                        ref_idx = nxt_idx 
                        ref_val = board[ref_idx]
                    else: # collide
                        if not(i == nxt_idx[0] and j == nxt_idx[1]):
                            spawn_flag = True
                        ref_idx = nxt_idx 
                        ref_val = val
                        board[i][j] = 0
                        board[ref_idx] = val
        if spawn_flag:
            self.spawn(1)

        done = self._done()

        reward = np.max(board) / 16.0 if done else 0 #1.0 * np.any(board >= 10) # 10 = 1024 ... TODO: change to 2048?
        # reward = np.max(board) / 10.0

        return self.state(), reward, done, {}

    def _render(self, mode='human', close=False):

        if close and self.viewer is not None:
            try:
                self.viewer.kill()
                self.viewer.terminate()
                self.viewer = None
                os.remove(self.disp_path)
            except OSError:
                pass
            return

        if self.viewer is None:
            if not os.path.exists(self.disp_path):
                os.mkfifo(self.disp_path)
            self.viewer = Popen(['xterm', '-e', 'tail -f %s' % self.disp_path], close_fds=True)

        with open(self.disp_path, 'w') as p:
            p.write(str(self) + '\n\n')

    def __repr__(self):
        return self.board.__repr__()
    def __str__(self):
        return self.board.__str__()
    def _done(self):
        board = self.board
        if not board.all():
            return False
        # check vert ...
        if not (board[1:] - board[:-1]).all():
            return False
        # check horz ...
        if not (board[:,1:] - board[:,:-1]).all():
            return False
        return True

def main():
    game = Game(4,4)
    #for action in (LEFT,RIGHT,UP,DOWN):
    #    print '=================='
    #    game.reset()
    #    game.spawn(6)
    #    print game
    #    print action_str[action]
    #    game.step(action)
    #    print game
    game.reset()
    print game.state()

    #print game.action_space.sample()
    #print game.observation_space.sample()

    done = False
    while not done:
        game.render()
        k = raw_input()

        if k == 'u':
            a = UP
        elif k == 'd':
            a = DOWN
        elif k == 'l':
            a = LEFT
        elif k == 'r':
            a = RIGHT
        else:
            continue

        state, reward, done, _  = game.step(a)

if __name__ == "__main__":
    main()
