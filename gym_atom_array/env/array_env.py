import gym
import gym.spaces as spaces
import numpy as np

from dataclasses import dataclass

ACTION_TO_DIFF = [(-1, 0), (1, 0), (0, -1), (0, 1)]


@dataclass
class Config:
    Render: bool = False

    # Environment Parameters
    FillFraction: float = 0.5
    TransitProb: float = 0.999
    GridTime: int = 50
    TweezerTime: int = 50

    # Rewards
    TimeMultiplier: float = 0.1
    DefaultPenalty: float = 0.1

    TargetRelease: float = 0  # Release into target
    ReservRelease: float = 0  # Release into reserve
    TargetPickUp: float = 0  # Pick up a target atom
    ReservPickUp: float = 0  # Pick up a reserve atom

    DuplicatePickUp: float = 0  # Pick up when atom already in MT
    DuplicateRelease: float = 0  # Release into trap with atom
    EmptyPickUp: float = 0  # Pick up from trap without atom
    EmptyRelease: float = 0  # Release when no atom in MT
    CollisionLoss: float = 0  # Collision
    MTCollLoss: float = 0  # Collision

    RePickUp: float = 0  # Pick up an atom that was just released
    UndoMove: float = 0  # Current move undoes the previous move


class ArrayEnv(gym.Env):
    def __init__(
        self, n_rows: int, n_cols: int, targets, config: Config, seed=0
    ) -> None:
        # State
        self._grid = np.array(
            [[0 for _ in range(n_cols)] for __ in range(n_rows)], dtype=np.int8
        )
        self._tgrid = np.array(
            [[0 for _ in range(n_cols)] for __ in range(n_rows)], dtype=np.int8
        )
        for tr, tc in targets:
            self._tgrid[tr][tc] = 1
        self._target_rewards = np.copy(self._tgrid)

        self._targets = set(targets)
        self._mt_pos = np.array((0, 0))
        self._mt_atom = False
        self._mt_atom_obs = 0
        self._move_len = 0
        self._total_time = 0

        # Config stuff
        self._all_targets = set(targets)
        self.n_rows = n_rows
        self.n_cols = n_cols
        self.config = config

        # Observations
        obs_grid = spaces.Box(low=0, high=1, shape=(n_rows, n_cols), dtype=np.int8)
        obs_tgrid = spaces.Box(low=0, high=1, shape=(n_rows, n_cols), dtype=np.int8)
        obs_mt_pos = spaces.MultiDiscrete([n_rows, n_cols])
        obs_mt_atom = spaces.Discrete(2)
        self.observation_space = spaces.Dict(
            {
                "grid": obs_grid,
                "tar_grid": obs_tgrid,
                "mt_position": obs_mt_pos,
                "mt_atom": obs_mt_atom,
            }
        )

        # Actions
        # self.action_space = spaces.MultiDiscrete([6, max(n_rows, n_cols)])
        self.action_space = spaces.Discrete(6)

        self.np_random = np.random.default_rng()
        self.reset()

    def _reset_target_rewards(self):
        self._target_rewards = np.copy(self._tgrid) * self.config.TargetRelease

    def _fill_grid(self):
        filled = 0
        while filled < len(self._all_targets):
            self._targets = set(self._all_targets)
            filled = 0
            self._grid = np.zeros(self._grid.shape, dtype=np.int8)
            fill_prob = self.config.FillFraction
            for i in range(self.n_rows):
                for j in range(self.n_cols):
                    if self.np_random.random() < fill_prob:
                        filled += 1
                        self._grid[i][j] = 1
                        self._targets.discard((i, j))

    def reset(self, *, seed=None, options=None):
        # super().reset(seed=seed, options=options)

        # Reset and fill grid
        self._reset_target_rewards()
        self._fill_grid()

        # Reset MT
        mt_r = self.np_random.integers(0, self.n_rows)
        mt_c = self.np_random.integers(0, self.n_cols)
        self._mt_pos = np.array((mt_r, mt_c))
        self._mt_atom = False
        self._mt_atom_obs = 0
        self._move_len = 0
        self._total_time = 0

        return {
            "grid": self._grid,
            "tar_grid": self._tgrid,
            "mt_position": self._mt_pos,
            "mt_atom": 1 * self._mt_atom,
        }

    def render(self, *args):
        for r in range(self.n_rows):
            for c in range(self.n_cols):
                print(self._grid[r][c], end="")
                print("*" if self._tgrid[r][c] == 1 else " ", end=" ")
            print()
        print(
            f"MT at {self._mt_pos}. Atom in: {self._mt_atom}, obs: {self._mt_atom_obs}"
        )
        print(f"Remaining targets: {self._targets}")

    def _check_cell(self, pos):
        r, c = pos
        return r > -1 and c > -1 and r < self.n_rows and c < self.n_cols

    def _clip_move(self, pos):
        return np.clip(pos, (0, 0), (self.n_rows, self.n_cols))

    def _step(self, action):
        dir = action
        pos = tuple(self._mt_pos)
        grid = self._grid
        config = self.config

        # Tweezer Pickup
        if dir == 4:
            if self._mt_atom:
                # Duplicate pick up
                return config.DuplicatePickUp, True
            if grid[pos] == 0:
                # Empty Pick up
                return config.EmptyPickUp, True

            self._mt_atom = True
            grid[pos] = 0
            self._total_time += config.TweezerTime

            if pos in self._all_targets:
                self._targets.add(pos)
                return config.TargetPickUp, False
            return config.ReservPickUp, False

        # Tweezer Release
        if dir == 5:
            if not self._mt_atom:
                # Empty release
                return config.EmptyRelease, True
            if grid[pos] == 1:
                # Duplicate release
                return config.DuplicateRelease, True

            self._mt_atom = False
            grid[pos] = 1
            self._total_time += config.TweezerTime

            if pos in self._targets:
                self._targets.discard(pos)
                temp = self._target_rewards[pos]
                self._target_rewards[pos] *= 0
                return temp, False
            return config.ReservRelease, False

        # Movement
        pos = self._mt_pos
        diff = ACTION_TO_DIFF[dir]
        reward = 0
        for _ in range(1):
            if not self._check_cell(pos + diff):
                break
            self._move_len += 1

            reward -= self.config.GridTime * self.config.TimeMultiplier
            self._total_time += self.config.GridTime

            pos += diff
            if self._mt_atom and grid[tuple(pos)] == 1:
                # Collision
                reward += self.config.CollisionLoss
                grid[pos] = 0
                reward += self.config.MTCollLoss if self._mt_atom else 0
                self._mt_atom = False

        return reward, False

    def _terminal_reward(self):
        return len(self._all_targets)  # - self._total_time * self.config.TimeMultiplier

    def step(self, action):
        term, trunc, reward = False, False, 0
        reward, stop = self._step(action)

        if len(self._targets) == 0:
            term = True
            reward += self._terminal_reward()

        if self.config.Render:
            self.render()
            print(self._targets)

        return (
            {
                "grid": self._grid,
                "tar_grid": self._tgrid,
                "mt_position": self._mt_pos,
                "mt_atom": 1 * self._mt_atom,
            },
            reward - self.config.DefaultPenalty,
            term or trunc,
            {},
        )


if __name__ == "__main__":
    small_grid = [
        (1, 1),
        (1, 2),
        (1, 3),
    ]  # (2, 1), (2, 2), (2, 3), (3, 1), (3, 2), (3, 3)]
    config = Config(Render=True)
    env = ArrayEnv(5, 5, small_grid, config)
    env.render()
    while True:
        while True:
            try:
                inp = int(input())
                if inp in range(5):
                    print("Dir: ", inp)
                    break
            except Exception as e:
                pass
        while True:
            try:
                delt = int(input())
                print("Delta: ", delt)
                break
            except Exception as e:
                pass
        act = (inp, delt)
        _, _, term, trunk, _ = env.step(act)
        if term or trunk:
            break
