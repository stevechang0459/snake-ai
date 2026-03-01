import sys
import pygame
import keyboard
import random
import torch
from sb3_contrib import MaskablePPO
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

# Import your custom environment
from snake_game_custom_wrapper_cnn import SnakeEnv

import warnings

# Filter out the specific FutureWarning related to torch.load and weights_only
# This is safe because we are loading our own trained models
warnings.filterwarnings("ignore", category=FutureWarning, module="stable_baselines3")

# Parameter Settings
NUM_ENVS = 1
FPS = 0                # Display speed
RENDER = True
RENDER_FEATURE_MAP = False

NUM_EPISODE = 10
ROUND_DELAY = 5         # unit: sec
HUGE_NEGATIVE = -1e8


if torch.backends.mps.is_available():
    MODEL_PATH = r"trained_models_cnn_mps/ppo_snake_final"
else:
    MODEL_PATH = r"C:\Users\steve\OneDrive\Workspace\Github\Python\snake-ai\main\PPO_Snake_Game_21x21_CNN_v6_20260224_082348\PPO_Snake_Game_21x21_CNN_v6_final_20260224_082348.zip"

activations = {}

def get_activation(layer_name):
    def hook(model, input, output):
        activations[layer_name] = output.detach().cpu().numpy()
    return hook

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

def tile_direct(features, rows, cols):
    _, H, W = features.shape
    # Reshape to split dimensions -> Transpose to swap rows and height -> Reshape again to flatten into a 2D large image
    return features.reshape(rows, cols, H, W).transpose(0, 2, 1, 3).reshape(rows * H, cols * W)

def tile_with_padding(features, rows, cols, padding=1):
    padded = np.pad(features, ((0, 0), (padding, padding), (padding, padding)), mode='constant', constant_values=np.nan)
    _, new_H, new_W = padded.shape
    return padded.reshape(rows, cols, new_H, new_W).transpose(0, 2, 1, 3).reshape(rows * new_H, cols * new_W)

def main():
    seed = random.randint(0, 1e9)
    print(f"Using seed = {seed} for testing.")

    # Initialize Vectorized Environment
    env = SnakeEnv(seed=seed, board_size=21, limit_step=True, silent_mode=not RENDER)

    # Load the trained model
    model = MaskablePPO.load(MODEL_PATH)
    cnn_extractor = model.policy.features_extractor.cnn
    cnn_extractor[0].register_forward_hook(get_activation('Conv1_42x42'))
    cnn_extractor[2].register_forward_hook(get_activation('Conv2_20x20'))
    cnn_extractor[4].register_forward_hook(get_activation('Conv3_9x9'))
    cnn_extractor[6].register_forward_hook(get_activation('Conv4_7x7'))

    total_reward = 0
    total_score = 0
    total_step = 0
    min_score = 1e9
    max_score = 0
    clock = pygame.time.Clock()

    if RENDER_FEATURE_MAP:
        # ================= Window Initialization =================
        mpl.rcParams['toolbar'] = 'None'
        plt.ion()

        invisible_color = '#010101'

        # Each window only needs 1 clean canvas (1x1 subplot)
        fig1, ax1 = plt.subplots(figsize=(10, 5), facecolor=invisible_color)
        fig2, ax2 = plt.subplots(figsize=(8, 8), facecolor=invisible_color)
        fig3, ax3 = plt.subplots(figsize=(7, 7), facecolor=invisible_color)
        fig4, ax4 = plt.subplots(figsize=(16, 8), facecolor=invisible_color)

        def setup_floating_window(fig, title):
            try:
                window = fig.canvas.manager.window
                window.overrideredirect(True)
                window.attributes('-transparentcolor', invisible_color)
                window.attributes('-topmost', True) # Keep it always floating on top

                # Dragging logic
                drag_state = {'dragging': False, 'x': 0, 'y': 0}
                def on_press(event):
                    if event.button == 1:
                        drag_state['dragging'] = True
                        drag_state['x'] = window.winfo_pointerx() - window.winfo_rootx()
                        drag_state['y'] = window.winfo_pointery() - window.winfo_rooty()
                def on_release(event):
                    if event.button == 1: drag_state['dragging'] = False
                def on_motion(event):
                    if drag_state['dragging']:
                        window.geometry(f"+{window.winfo_pointerx()-drag_state['x']}+{window.winfo_pointery()-drag_state['y']}")

                fig.canvas.mpl_connect('button_press_event', on_press)
                fig.canvas.mpl_connect('button_release_event', on_release)
                fig.canvas.mpl_connect('motion_notify_event', on_motion)
            except: pass

        setup_floating_window(fig1, "Conv1")
        setup_floating_window(fig2, "Conv2")
        setup_floating_window(fig3, "Conv3")
        setup_floating_window(fig4, "Conv4")

        # my_cmap = mpl.colormaps['viridis'].copy()
        # my_cmap.set_bad(color='#555555')

        # Prepare 4 single image objects (insert a 1x1 matrix as placeholder first, dynamically replace with the full image later)
        im1 = ax1.imshow(np.zeros((1, 1)), cmap='viridis', aspect='auto')
        im2 = ax2.imshow(np.zeros((1, 1)), cmap='viridis', aspect='auto')
        im3 = ax3.imshow(np.zeros((1, 1)), cmap='viridis', aspect='auto')
        im4 = ax4.imshow(np.zeros((1, 1)), cmap='viridis', aspect='auto')

        for ax in [ax1, ax2, ax3, ax4]:
            ax.axis('off')

        ax1.set_title("Conv1 (32 ch)", color='white', fontsize=12, pad=5)
        ax2.set_title("Conv2 (64 ch)", color='white', fontsize=12, pad=5)
        ax3.set_title("Conv3 (64 ch)", color='white', fontsize=12, pad=5)
        ax4.set_title("Conv4 (128 ch)", color='white', fontsize=12, pad=5)

        for fig in [fig1, fig2, fig3, fig4]:
            fig.subplots_adjust(left=0.01, right=0.92, bottom=0.01, top=0.93)

        def create_cbar(fig, im):
            cbar_ax = fig.add_axes([0.94, 0.05, 0.02, 0.9])
            cbar = fig.colorbar(im, cax=cbar_ax)
            cbar_ax.set_facecolor('black') # Set a black background for the colorbar to keep values readable
            cbar_ax.yaxis.set_tick_params(color='white', labelcolor='white')
            return cbar

        cbar1 = create_cbar(fig1, im1)
        cbar2 = create_cbar(fig2, im2)
        cbar3 = create_cbar(fig3, im3)
        cbar4 = create_cbar(fig4, im4)

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

            if RENDER_FEATURE_MAP:
                # ================= Update Window Data =================
                if 'Conv1_42x42' in activations and 'Conv2_20x20' in activations and \
                    'Conv3_9x9' in activations and 'Conv4_7x7' in activations:

                    # Extract feature maps
                    f1 = activations['Conv1_42x42'][0]
                    f2 = activations['Conv2_20x20'][0]
                    f3 = activations['Conv3_9x9'][0]
                    f4 = activations['Conv4_7x7'][0]

                    # Tile using NumPy (Layout: 4x8, 8x8, 8x8, 8x16)
                    vmin1, vmax1 = f1.min(), f1.max() + 1e-8
                    vmin2, vmax2 = f2.min(), f2.max() + 1e-8
                    vmin3, vmax3 = f3.min(), f3.max() + 1e-8
                    vmin4, vmax4 = f4.min(), f4.max() + 1e-8

                    PADDING = 1
                    grid1 = tile_with_padding(f1, 4, 8, padding=PADDING)
                    grid2 = tile_with_padding(f2, 8, 8, padding=PADDING)
                    grid3 = tile_with_padding(f3, 8, 8, padding=PADDING)
                    grid4 = tile_with_padding(f4, 8, 16, padding=PADDING)

                    # Calculate Max/Min for each large image and update the single canvas
                    im1.set_data(grid1)
                    im1.set_clim(vmin=vmin1, vmax=vmax1)

                    im2.set_data(grid2)
                    im2.set_clim(vmin=vmin2, vmax=vmax2)

                    im3.set_data(grid3)
                    im3.set_clim(vmin=vmin3, vmax=vmax3)

                    im4.set_data(grid4)
                    im4.set_clim(vmin=vmin4, vmax=vmax4)

                    # Synchronize Colorbar and refresh windows
                    cbar1.update_normal(im1)
                    cbar2.update_normal(im2)
                    cbar3.update_normal(im3)
                    cbar4.update_normal(im4)

                    fig1.canvas.draw_idle()
                    fig1.canvas.flush_events()
                    fig2.canvas.draw_idle()
                    fig2.canvas.flush_events()
                    fig3.canvas.draw_idle()
                    fig3.canvas.flush_events()
                    fig4.canvas.draw_idle()
                    fig4.canvas.flush_events()

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
