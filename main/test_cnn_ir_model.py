import sys
import pygame
import keyboard
import random
import numpy as np
import openvino.runtime as ov

# Import your custom environment
from snake_game_custom_wrapper_cnn import SnakeEnv

# Parameter Settings
NUM_ENVS = 1
FPS = 30                # Display speed
RENDER = True

NUM_EPISODE = 10
ROUND_DELAY = 5         # unit: sec
HUGE_NEGATIVE = -1e8

IR_MODEL_PATH = f"model_ir/snake_fp16_b{NUM_ENVS}.xml"

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

    # Initialize OpenVINO Core
    core = ov.Core()
    devices = core.available_devices
    device_name = "NPU" if "NPU" in devices else "CPU"
    print(f"Targeting device: {device_name}")

    try:
        # Load and compile the OpenVINO model
        compiled_model = core.compile_model(IR_MODEL_PATH, device_name=device_name)
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    infer_request = compiled_model.create_infer_request()
    input_layer = compiled_model.input(0)
    output_layer = compiled_model.output(0)

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

            # Transpose observations from (H, W, C) to (C, H, W) for ONNX/OpenVINO
            obs_chw = np.transpose(obs, (2, 0, 1))
            # (C, H, W) -> (1, C, H, W)
            input_tensor = np.expand_dims(obs_chw, axis=0).astype(np.float32)

            # Run Inference
            results = infer_request.infer({input_layer: input_tensor})
            logits = results[output_layer]

            # Process Action Masks
            action_mask = env.get_action_mask()
            if action_mask.ndim == 1:
                # (4,) -> (1, 4)
                action_mask = np.expand_dims(action_mask, axis=0)

            # Apply mask by setting invalid actions to a huge negative value
            logits[~action_mask] = HUGE_NEGATIVE

            # Select Action (Argmax) across axis 1 (the 4 possible actions)
            action = np.argmax(logits, axis=1)

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
                    last_action = ["UP", "LEFT", "RIGHT", "DOWN"][action.item()]
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
