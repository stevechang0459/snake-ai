import math
import cv2
import gym
import numpy as np

from snake_game import SnakeGame

class SnakeEnv(gym.Env):
    def __init__(self, seed=0, board_size=12, limit_step=True, silent_mode=True, multi_process_render=False):
        super().__init__()
        self.game = SnakeGame(seed=seed, board_size=board_size, silent_mode=silent_mode, multi_process_render=multi_process_render)
        self.game.reset()

        self.silent_mode = silent_mode
        if board_size == 12:
            self.obs_resolution = 84
        else:
            self.obs_resolution = board_size * 8

        self.action_space = gym.spaces.Discrete(4) # 0: UP, 1: LEFT, 2: RIGHT, 3: DOWN
        self.observation_space = gym.spaces.Box(
            low=0, high=255,
            shape=(self.obs_resolution, self.obs_resolution, 3),
            dtype=np.uint8
        )

        self.board_size = board_size
        self.grid_size = board_size ** 2 # Max length of snake is board_size^2
        self.init_snake_size = len(self.game.snake)
        self.max_growth = self.grid_size - self.init_snake_size

        self.done = False

        if limit_step:
            # More than enough steps to get the food.
            # self.step_limit = self.grid_size * 4
            self.step_limit = self.grid_size * 2
        else:
            self.step_limit = 1e9 # Basically no limit.
        self.reward_step_counter = 0

    def reset(self):
        self.game.reset()
        self.done = False
        self.reward_step_counter = 0
        obs = self._generate_observation()
        return obs

    def step(self, action):
        self.done, info = self.game.step(action) # info = {"snake_size": int, "snake_head_pos": np.array, "prev_snake_head_pos": np.array, "food_pos": np.array, "food_obtained": bool}
        obs = self._generate_observation()

        reward = 0.0
        self.reward_step_counter += 1

        # Victory Reward
        if info["snake_size"] == self.grid_size: # Snake fills up the entire board. Game over.
            reward = self.max_growth * 0.1 # Victory reward
            self.done = True
            if not self.silent_mode:
                self.game.sound_victory.play()
            return obs, reward, self.done, info

        # Timeout
        if self.reward_step_counter > self.step_limit: # Step limit reached, game over.
            self.reward_step_counter = 0
            self.done = True

        # Game Over
        if self.done: # Snake bumps into wall or itself. Episode is over.
            # Game Over penalty is based on snake size.
            # reward = - math.pow(self.max_growth, (self.grid_size - info["snake_size"]) / self.max_growth) # (-max_growth, -1)
            # reward = reward * 0.1
            base_penalty = 1.0
            penalty_scale = info["snake_size"] / self.grid_size
            # reward = - (base_penalty + 4.0 * penalty_scale)
            reward = - (base_penalty + 4.0 * penalty_scale) ** 2
            return obs, reward, self.done, info

        # Food Eaten
        elif info["food_obtained"]: # Food eaten. Reward boost on snake size.
            # reward = max(info["snake_size"] / self.grid_size, self.board_size / info["snake_size"])
            # reward = 1.0 + (info["snake_size"] / self.grid_size)
            reward = (1.0 + (info["snake_size"] / self.grid_size)) ** 2
            self.reward_step_counter = 0 # Reset reward step counter

        else:
            # Survival Penalty
            step_penalty = - (0.1 / info["snake_size"])
            # Give a tiny reward/penalty to the agent based on whether it is heading towards the food or not.
            # Not competing with game over penalty or the food eaten reward.

            # Navigation Reward
            if np.linalg.norm(info["snake_head_pos"] - info["food_pos"]) < np.linalg.norm(info["prev_snake_head_pos"] - info["food_pos"]):
                navigation_reward = 1 / info["snake_size"]
            else:
                navigation_reward = - 1 / info["snake_size"]
            reward = step_penalty + navigation_reward
            reward = reward * 0.1

        # max_score ~= 286.0 + 438 * 0.1 + 0.5 = 330.3
        # min_score ~= ((-0.1 / 3) * 0.1 * 882) + -(1.0 + 4.0 * (3/441)) = -3.97

        return obs, reward, self.done, info

    def render(self):
        self.game.render()

    # This is called Action Masking in sb3_contrib's MaskablePPO algorithm.
    def get_action_mask(self):
        return np.array([[self._check_action_validity(a) for a in range(self.action_space.n)]])

    # Check if the action is against the current direction of the snake or is ending the game.
    def _check_action_validity(self, action):
        current_direction = self.game.direction
        snake_list = self.game.snake
        row, col = snake_list[0]
        if action == 0: # UP
            if current_direction == "DOWN":
                return False
            else:
                row -= 1

        elif action == 1: # LEFT
            if current_direction == "RIGHT":
                return False
            else:
                col -= 1

        elif action == 2: # RIGHT
            if current_direction == "LEFT":
                return False
            else:
                col += 1

        elif action == 3: # DOWN
            if current_direction == "UP":
                return False
            else:
                row += 1

        # Check if snake collided with itself or the wall. Note that the tail of the snake would be popped if the snake did not eat food in the current step.
        if (row, col) == self.game.food:
            game_over = (
                (row, col) in snake_list # The snake won't pop the last cell if it ate food.
                or row < 0
                or row >= self.board_size
                or col < 0
                or col >= self.board_size
            )
        else:
            game_over = (
                (row, col) in snake_list[:-1] # The snake will pop the last cell if it did not eat food.
                or row < 0
                or row >= self.board_size
                or col < 0
                or col >= self.board_size
            )

        if game_over:
            return False
        else:
            return True

    # EMPTY: BLACK; SnakeBODY: GRAY; SnakeHEAD: GREEN; FOOD: RED;
    def _generate_observation(self):
        obs = np.zeros((self.game.board_size, self.game.board_size), dtype=np.uint8)

        # Set the snake body to gray with linearly decreasing intensity from head to tail.
        obs[tuple(np.transpose(self.game.snake))] = np.linspace(200, 50, len(self.game.snake), dtype=np.uint8)

        # Stack single layer into 3-channel-image.
        obs = np.stack((obs, obs, obs), axis=-1)

        # Set the snake head to green and the tail to blue
        obs[tuple(self.game.snake[0])] = [0, 255, 0]
        obs[tuple(self.game.snake[-1])] = [255, 0, 0]

        # Set the food to red
        obs[self.game.food] = [0, 0, 255]

        # Enlarge the observation to 84x84
        # obs = np.repeat(np.repeat(obs, 7, axis=0), 7, axis=1)
        obs = cv2.resize(obs, (self.obs_resolution, self.obs_resolution), interpolation=cv2.INTER_NEAREST)

        return obs

    def get_frame(self):
        return self.game.get_frame()

# Test the environment using random actions
# NUM_EPISODES = 100
# RENDER_DELAY = 0.001
# from matplotlib import pyplot as plt

# if __name__ == "__main__":
#     env = SnakeEnv(silent_mode=False)

    # # Test Init Efficiency
    # print(MODEL_PATH_S)
    # print(MODEL_PATH_L)
    # num_success = 0
    # for i in range(NUM_EPISODES):
    #     num_success += env.reset()
    # print(f"Success rate: {num_success/NUM_EPISODES}")

    # sum_reward = 0

    # # 0: UP, 1: LEFT, 2: RIGHT, 3: DOWN
    # action_list = [1, 1, 1, 0, 0, 0, 2, 2, 2, 3, 3, 3]

    # for _ in range(NUM_EPISODES):
    #     obs = env.reset()
    #     done = False
    #     i = 0
    #     while not done:
    #         plt.imshow(obs, interpolation='nearest')
    #         plt.show()
    #         action = env.action_space.sample()
    #         # action = action_list[i]
    #         i = (i + 1) % len(action_list)
    #         obs, reward, done, info = env.step(action)
    #         sum_reward += reward
    #         if np.absolute(reward) > 0.001:
    #             print(reward)
    #         env.render()

    #         time.sleep(RENDER_DELAY)
    #     # print(info["snake_length"])
    #     # print(info["food_pos"])
    #     # print(obs)
    #     print("sum_reward: %f" % sum_reward)
    #     print("episode done")
    #     # time.sleep(100)

    # env.close()
    # print("Average episode reward for random strategy: {}".format(sum_reward/NUM_EPISODES))
