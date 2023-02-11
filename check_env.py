from gym_atom_array.env import ArrayEnv, Config, get_action_mask

from gym.utils.env_checker import check_env as gym_check

size = 5
ROWS, COLS = size, size
small_grid = [(1, 1), (1, 2), (1, 3)]
config = Config(Render=True)

env = ArrayEnv(n_rows=ROWS, n_cols=COLS, targets=small_grid, config=config)

gym_check(env)

obs = env.reset()
env.render()
print(get_action_mask(obs))
