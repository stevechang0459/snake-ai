import sys
import pygame
import keyboard
import random
import numpy as np
import torch
from sb3_contrib import MaskablePPO
from stable_baselines3.common.vec_env import SubprocVecEnv

# Import your custom environment
from snake_game_custom_wrapper_cnn import SnakeEnv

import warnings

# Filter out the specific FutureWarning related to torch.load and weights_only
# This is safe because we are loading our own trained models
warnings.filterwarnings("ignore", category=FutureWarning, module="stable_baselines3")

# Parameter Settings
GRID_COLS = 4
GRID_ROWS = 2
NUM_ENVS = GRID_COLS * GRID_ROWS        # Run n environments simultaneously
GAP_RATIO = 0.02                        # Gap takes up 2% of screen width
FPS = 30                                # Display speed
SCREEN_RATIO = 1
RENDER = True

NUM_EPISODE = 10
HUGE_NEGATIVE = -1e8

if torch.backends.mps.is_available():
    MODEL_PATH = r"trained_models_cnn_mps/ppo_snake_final"
else:
    MODEL_PATH = r"trained_models_cnn_v3_20260217_012656/ppo_snake_final_v3_20260217_012656"

def make_env(seed=0):
    def _init():
        # Set multi_process_render based on RENDER setting
        env = SnakeEnv(seed=seed, limit_step=True, silent_mode=True, multi_process_render=RENDER)
        return env
    return _init

def main():
    seed = random.randint(0, 1e9)
    print(f"Using seed = {seed} for testing.")

    # Initialize Vectorized Environment
    env = SubprocVecEnv([make_env(seed=seed + i) for i in range(NUM_ENVS)])

    # Load the trained model
    sb3_model = MaskablePPO.load(MODEL_PATH)

    # Initialize Pygame
    pygame.init()

    if RENDER:
        screen_info = pygame.display.Info()
        screen_w = int(screen_info.current_w * SCREEN_RATIO)
        screen_h = int(screen_info.current_h * SCREEN_RATIO)

        # Get original frame dimensions (H, W, C) from environment
        frames = env.env_method("get_frame")[0]
        orig_w = frames.shape[0]
        orig_h = frames.shape[1]
        print(f"Original frame size: {orig_w}x{orig_h}")

        # Calculate scaling to fit the grid on screen
        scale_w = (screen_w * (1 - GAP_RATIO)) / (GRID_COLS * orig_w)
        scale_h = (screen_h * (1 - GAP_RATIO)) / (GRID_ROWS * orig_h)
        final_scale = min(scale_w, scale_h)

        display_w = int(orig_w * final_scale)
        display_h = int(orig_h * final_scale)

        # Calculate gaps to absorb remaining black borders
        gap_x = (screen_w - (display_w * GRID_COLS)) // (GRID_COLS + 1)
        gap_y = (screen_h - (display_h * GRID_ROWS)) // (GRID_ROWS + 1)

        screen = pygame.display.set_mode((screen_w, screen_h), pygame.NOFRAME)
    else:
        # Initialize timer for FPS printing
        last_print_time = pygame.time.get_ticks()

    clock = pygame.time.Clock()
    obs = env.reset() # Get initial observations for n environments
    running = True

    print("Starting execution loop...")
    while running:
        if RENDER:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE or event.key == pygame.K_q:
                        running = False
        else:
            if keyboard.is_pressed('esc') or keyboard.is_pressed('q'):
                print("Exit command detected. Closing game...")
                running = False

        action_masks = np.array(env.env_method("get_action_mask"))
        action, _ = sb3_model.predict(obs, action_masks=action_masks, deterministic=True)

        # Execute step in environments
        obs, _, _, _ = env.step(action)

        if RENDER:
            screen.fill((0, 0, 0))
            # Get frames from all sub-processes for rendering
            frames = env.env_method("get_frame")

            for i, frame in enumerate(frames):
                surf = pygame.surfarray.make_surface(frame)

                # Smoothscaling for better quality, use pygame.transform.scale for performance
                scaled_surf = pygame.transform.smoothscale(surf, (display_w, display_h))

                # Calculate grid position for blitting
                col, row = i % GRID_COLS, i // GRID_COLS
                x = gap_x + col * (display_w + gap_x)
                y = gap_y + row * (display_h + gap_y)
                screen.blit(scaled_surf, (x, y))

            current_fps = clock.get_fps()
            fps_text = pygame.font.SysFont("Arial", 24).render(f"FPS: {current_fps:.1f}", True, (255, 255, 0))
            screen.blit(fps_text, (10, 10))

            # Update display
            pygame.display.flip()
            # Control loop timing
            clock.tick(FPS) # Uncomment to cap FPS
            # clock.tick()
        else:
            # Console output for performance monitoring
            current_time = pygame.time.get_ticks()
            if current_time - last_print_time >= 2000:
                current_fps = clock.get_fps()
                print(f"FPS: {current_fps:7.3f}")
                last_print_time = current_time
            clock.tick()

    print("Shutting down...")
    env.close()
    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()
