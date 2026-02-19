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
FPS = 30                # Display speed
RENDER = True

NUM_EPISODE = 10
ROUND_DELAY = 5         # unit: sec
HUGE_NEGATIVE = -1e8

if torch.backends.mps.is_available():
    MODEL_PATH = r"trained_models_cnn_mps/ppo_snake_final"
else:
    MODEL_PATH = r"PPO_Snake_Game_21x21_CNN_v5_20260219_175719/ppo_snake_9000000_steps"

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

    # ================= 視窗初始化 =================
    import matplotlib as mpl
    mpl.rcParams['toolbar'] = 'None'
    plt.ion()

    invisible_color = '#010101'

    # --- 視窗一：Conv1 (4x8 = 32 Channels) ---
    fig1, axes1 = plt.subplots(4, 8, figsize=(10, 5), facecolor=invisible_color)
    # --- 視窗二：Conv2 (8x8 = 64 Channels) ---
    fig2, axes2 = plt.subplots(8, 8, figsize=(8, 8), facecolor=invisible_color)
    # --- 視窗三：Conv3 (8x8 = 64 Channels，解析度 9x9) ---
    fig3, axes3 = plt.subplots(8, 8, figsize=(7, 7), facecolor=invisible_color)
    # --- 視窗四：Conv4 (8x16 = 128 Channels，解析度 7x7) ---
    fig4, axes4 = plt.subplots(8, 16, figsize=(6, 6), facecolor=invisible_color)

    def setup_floating_window(fig, title):
        try:
            window = fig.canvas.manager.window
            window.overrideredirect(True)
            window.attributes('-transparentcolor', invisible_color)
            window.attributes('-topmost', True) # 讓它永遠漂浮在最上層

            # 拖曳邏輯
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

    # 準備圖片物件
    ims1 = [ax.imshow(np.zeros((42, 42)), cmap='viridis') for ax in axes1.flat]
    for ax in axes1.flat: ax.axis('off')

    ims2 = [ax.imshow(np.zeros((20, 20)), cmap='viridis') for ax in axes2.flat]
    for ax in axes2.flat: ax.axis('off')

    ims3 = [ax.imshow(np.zeros((9, 9)), cmap='viridis') for ax in axes3.flat]
    for ax in axes3.flat: ax.axis('off')

    ims4 = [ax.imshow(np.zeros((7, 7)), cmap='viridis') for ax in axes4.flat]
    for ax in axes4.flat: ax.axis('off')

    # fig.subplots_adjust(0.01, 0.01, 0.99, 0.99, 0.01, 0.01)
    fig1.subplots_adjust(left=0.01, right=0.92, bottom=0.01, top=0.99, wspace=0.01, hspace=0.01)
    fig2.subplots_adjust(left=0.01, right=0.92, bottom=0.01, top=0.99, wspace=0.01, hspace=0.01)
    fig3.subplots_adjust(left=0.01, right=0.92, bottom=0.01, top=0.99, wspace=0.01, hspace=0.01)
    fig4.subplots_adjust(left=0.01, right=0.92, bottom=0.01, top=0.99, wspace=0.01, hspace=0.01)

    # Colorbar 建議保留背景，不然字會看不清楚
    cbar1_ax = fig1.add_axes([0.94, 0.05, 0.02, 0.9])
    cbar1 = fig1.colorbar(ims1[0], cax=cbar1_ax)
    cbar1_ax.set_facecolor('black') # 讓 Colorbar 有個黑底比較好讀數值
    cbar1_ax.yaxis.set_tick_params(color='white', labelcolor='white')

    cbar2_ax = fig2.add_axes([0.94, 0.05, 0.02, 0.9])
    cbar2 = fig2.colorbar(ims2[0], cax=cbar2_ax)
    cbar2_ax.set_facecolor('black')
    cbar2_ax.yaxis.set_tick_params(color='white', labelcolor='white')

    cbar3_ax = fig3.add_axes([0.94, 0.05, 0.02, 0.9])
    cbar3 = fig3.colorbar(ims3[0], cax=cbar3_ax)
    cbar3_ax.set_facecolor('black')
    cbar3_ax.yaxis.set_tick_params(color='white', labelcolor='white')

    cbar4_ax = fig4.add_axes([0.94, 0.05, 0.02, 0.9])
    cbar4 = fig4.colorbar(ims4[0], cax=cbar4_ax)
    cbar4_ax.set_facecolor('black')
    cbar4_ax.yaxis.set_tick_params(color='white', labelcolor='white')

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

            # ================= 更新視窗資料 =================
            # if 'Conv1_42x42' in activations and 'Conv2_20x20' in activations:
            if 'Conv1_42x42' in activations and 'Conv2_20x20' in activations and \
                'Conv3_9x9' in activations and 'Conv4_7x7' in activations:
                fmap_conv1 = activations['Conv1_42x42'][0]
                fmap_conv2 = activations['Conv2_20x20'][0]
                fmap_conv3 = activations['Conv3_9x9'][0]
                fmap_conv4 = activations['Conv4_7x7'][0]

                # 各自獨立計算最高與最低活化值
                global_min1 = fmap_conv1.min()
                global_max1 = fmap_conv1.max() + 1e-8

                global_min2 = fmap_conv2.min()
                global_max2 = fmap_conv2.max() + 1e-8

                global_min3 = fmap_conv3.min()
                global_max3 = fmap_conv3.max() + 1e-8

                global_min4 = fmap_conv4.min()
                global_max4 = fmap_conv4.max() + 1e-8

                # 更新 Conv1
                for i, im in enumerate(ims1):
                    im.set_data(fmap_conv1[i])
                    im.set_clim(vmin=global_min1, vmax=global_max1)

                # 更新 Conv2
                for i, im in enumerate(ims2):
                    im.set_data(fmap_conv2[i])
                    im.set_clim(vmin=global_min2, vmax=global_max2)

                # 更新 Conv3
                for i, im in enumerate(ims3):
                    im.set_data(fmap_conv3[i])
                    im.set_clim(vmin=global_min3, vmax=global_max3)

                # 更新 Conv4
                for i, im in enumerate(ims4):
                    im.set_data(fmap_conv4[i])
                    im.set_clim(vmin=global_min4, vmax=global_max4)

                cbar1.update_normal(ims1[0])
                cbar2.update_normal(ims2[0])
                cbar3.update_normal(ims3[0])
                cbar4.update_normal(ims4[0])

                # 刷新視窗
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
