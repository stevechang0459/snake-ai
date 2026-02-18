import sys
import pygame
import keyboard
import random
import torch
from sb3_contrib import MaskablePPO

# Import your custom environment
from snake_game_custom_wrapper_cnn import SnakeEnv

import warnings

# Filter out the specific FutureWarning related to torch.load and weights_only
# This is safe because we are loading our own trained models
warnings.filterwarnings("ignore", category=FutureWarning, module="stable_baselines3")

# Parameter Settings
NUM_ENVS = 1
FPS = 30                # Display speed
RENDER = True

NUM_EPISODE = 10
ROUND_DELAY = 5         # unit: sec
HUGE_NEGATIVE = -1e8

if torch.backends.mps.is_available():
    MODEL_PATH = r"trained_models_cnn_mps/ppo_snake_final"
else:
    MODEL_PATH = r"trained_models_cnn_v3_20260217_012656/ppo_snake_final_v3_20260217_012656"

def responsive_wait(ms):
    if not RENDER:
        return
    clock = pygame.time.Clock()
    target_time = pygame.time.get_ticks() + ms
    while pygame.time.get_ticks() < target_time:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE or event.key == pygame.K_q:
                    pygame.quit()
                    sys.exit()
        pygame.display.flip()
        clock.tick(FPS)

def main():
    seed = random.randint(0, 1e9)
    print(f"Using seed = {seed} for testing.")

    # Initialize Vectorized Environment
    env = SnakeEnv(seed=seed, limit_step=True, silent_mode=not RENDER)

    # Load the trained model
    model = MaskablePPO.load(MODEL_PATH)

    total_reward = 0
    total_score = 0
    total_step = 0
    min_score = 1e9
    max_score = 0
    clock = pygame.time.Clock()

    for episode in range(NUM_EPISODE):
        obs = env.reset()
        episode_reward = 0
        done = False
        num_step = 0
        info = None
        sum_step_reward = 0
        print(f"=================== Episode {episode + 1} ==================")
        while not done:
            if keyboard.is_pressed('esc') or keyboard.is_pressed('q'):
                pygame.quit()
                sys.exit()

            # Run Inference
            action, _ = model.predict(obs, action_masks=env.get_action_mask(), deterministic=True)

            # Execute step in environments
            obs, reward, done, info = env.step(action)
            num_step += 1

            if RENDER:
                current_fps = clock.get_fps()

            if done:
                if info["snake_size"] == env.game.grid_size:
                    current_fps = clock.get_fps()
                    print(f"Food obtained at step {num_step:4d}, Food Reward: {reward:7.5f}, Step Reward: {sum_step_reward: 7.5f}, Snake Size: {info['snake_size']:3d}, FPS: {current_fps:6.3f}")
                    print(f"You are BREATHTAKING! Victory reward: {reward:7.5f}")
                    snake_size = info["snake_size"]
                else:
                    last_action = ["UP", "LEFT", "RIGHT", "DOWN"][action]
                    print(f"Gameover Penalty: {reward: 7.5f}, Last action: {last_action}")
                    # "Since the step function removes the tail when no food is
                    # eaten, the snake's length effectively decreases by 1."
                    snake_size = info["snake_size"] + 1
            elif info["food_obtained"]:
                if RENDER:
                    print(f"Food obtained at step {num_step:4d}, Food Reward: {reward:7.5f}, Step Reward: {sum_step_reward: 7.5f}, Snake Size: {info['snake_size']:3d}, FPS: {current_fps:6.3f}")
                sum_step_reward = 0
            else:
                sum_step_reward += reward

            episode_reward += reward
            if RENDER:
                env.render()
                # screen = pygame.display.get_surface()
                # fps_surface = fps_font.render(f"FPS: {current_fps:5.2f}", True, (255, 255, 0))
                # screen.blit(fps_surface, (10, 10))
                # pygame.display.flip()
                # Control loop timing
                clock.tick(FPS)
            else:
                clock.tick()

        episode_score = env.game.score
        if episode_score < min_score:
            min_score = episode_score
        if episode_score > max_score:
            max_score = episode_score

        print(f"Episode {episode + 1:2d}: Reward Sum: {episode_reward:7.5f}, Score: {episode_score:4d}, Total Steps: {num_step:4d}, Snake Size: {snake_size:3d}")
        total_step += num_step
        total_reward += episode_reward
        total_score += env.game.score
        if RENDER:
            print(f"Wait {ROUND_DELAY} seconds for next episode...")
            responsive_wait(ROUND_DELAY * 1000)

    env.close()
    print(f"=================== Summary ==================")
    print(f"Min Score: {min_score:4d}, Max Score: {max_score:4d}, Average Score: {total_score / NUM_EPISODE:7.5f}, Average Step: {total_step / NUM_EPISODE:8.3f}, Average reward: {total_reward / NUM_EPISODE:7.5f}")

if __name__ == "__main__":
    main()
