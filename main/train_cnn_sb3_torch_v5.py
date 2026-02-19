import os
import sys
import random
import torch
import torch.nn as nn
import keyboard  # Requires installation: pip install keyboard
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.vec_env import VecTransposeImage
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from snake_game_custom_wrapper_cnn import SnakeEnv
from datetime import datetime

# ==========================================
# 1. Utility Classes
# ==========================================

class DualLogger:
    """
    Writes output to both the terminal and a log file simultaneously.
    Acts like the Unix 'tee' command.
    """
    def __init__(self, filepath):
        self.terminal = sys.stdout
        self.log = open(filepath, "w", encoding='utf-8')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        # Flush both streams to ensure real-time logging (crucial for progress bars)
        self.terminal.flush()
        self.log.flush()

    def close(self):
        self.log.close()

class KeyboardStopCallback(BaseCallback):
    """
    Callback to stop training when a specific key is pressed.
    This bypasses the issue where Ctrl+C is sometimes ignored in Windows multi-process environments.
    """
    def __init__(self, key='q', verbose=1):
        super().__init__(verbose)
        self.key = key
        self.stopped = False

    def _on_step(self) -> bool:
        # Check if the key is pressed
        if keyboard.is_pressed(self.key) and not self.stopped:
            print(f"\n[KeyboardStop] Detected '{self.key}' key press! Stopping training safely...")
            self.stopped = True
            return False  # Returning False tells SB3 to stop training
        return True

class CustomCNN(BaseFeaturesExtractor):
    """
    Custom Convolutional Neural Network for larger input resolutions (e.g., 168x168).
    It uses 4 convolutional layers to ensure a large enough receptive field
    so the agent can "see" the entire board and understand global context.
    """
    def __init__(self, observation_space, features_dim=512):
        super().__init__(observation_space, features_dim)
        n_input_channels = observation_space.shape[0]

        # Define the 4-layer CNN architecture
        self.cnn = nn.Sequential(
            # Conv1: 168x168 -> 42x42, (168 - 8 + 2 * 2) = 164, 164 / 4 + 1 = 42
            nn.Conv2d(n_input_channels, 32, kernel_size=8, stride=4, padding=2),
            nn.ReLU(),
            # Conv2: 42x42 -> 20x20, (42 - 4 + 0) = 38, 38 / 2 + 1 = 20
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            # Extra downsampling stride
            # Conv3: 20x20 -> 9x9, (20 - 4 + 0) = 16, 16 / 2 + 1 = 9
            nn.Conv2d(64, 64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            # Extract deep features
            # Conv4: 9x9 -> 7x7, (9 - 3 + 0) = 6, 6 / 1 + 1 = 7
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Compute the flattened output size automatically dynamically by doing one forward pass
        with torch.no_grad():
            n_flatten = self.cnn(
                torch.as_tensor(observation_space.sample()[None]).float()
            ).shape[1]

        self.linear = nn.Sequential(
            nn.Linear(n_flatten, features_dim),
            nn.ReLU()
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.linear(self.cnn(observations))

# ==========================================
# 2. Global Configuration
# ==========================================

if torch.backends.mps.is_available():
    NUM_ENV = 32 * 2
else:
    NUM_ENV = 32
LOG_DIR = "logs"
N_STEPS = 2048
VER_NUM = "5"
BOARD_SIZE = 21

os.makedirs(LOG_DIR, exist_ok=True)

# ==========================================
# 3. Helper Functions
# ==========================================

# Linear scheduler for learning rate and clip range
def linear_schedule(initial_value, final_value=0.0):
    if isinstance(initial_value, str):
        initial_value = float(initial_value)
        final_value = float(final_value)
        assert (initial_value > 0.0)

    def scheduler(progress):
        return final_value + progress * (initial_value - final_value)

    return scheduler

def make_env(seed=0):
    def _init():
        # Note: board_size is set to 21 here as per your request
        env = SnakeEnv(seed=seed, board_size=BOARD_SIZE, limit_step=True)
        env = ActionMasker(env, SnakeEnv.get_action_mask)
        env = Monitor(env)
        env.seed(seed)
        return env
    return _init

# ==========================================
# 4. Main Training Loop
# ==========================================

def main():
    start_time = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Generate a list of unique random seeds for each environment
    seed_set = set()
    while len(seed_set) < NUM_ENV:
        seed_set.add(random.randint(0, int(1e9)))

    # Create the vectorized Snake environment (Multi-process)
    env = SubprocVecEnv([make_env(seed=s) for s in seed_set])
    env = VecTransposeImage(env)

    # Apply the custom CNN policy arguments
    policy_kwargs = dict(
        features_extractor_class=CustomCNN,
        features_extractor_kwargs=dict(features_dim=512),
    )

    # Instantiate PPO agent
    if torch.backends.mps.is_available():
        print(f"Using Device: MPS (Mac Metal)")
        lr_schedule = linear_schedule(5e-4, 2.5e-6)
        clip_range_schedule = linear_schedule(0.150, 0.025)
        model = MaskablePPO(
            "CnnPolicy",
            env,
            device="mps",
            verbose=1,
            n_steps=2048,
            batch_size=512*8,
            n_epochs=4,
            gamma=0.94,
            learning_rate=lr_schedule,
            clip_range=clip_range_schedule,
            tensorboard_log=LOG_DIR
        )
    else:
        print(f"Using Device: CUDA (NVIDIA)")
        lr_schedule = linear_schedule(2.5e-4, 2.5e-6)
        clip_range_schedule = linear_schedule(0.150, 0.025)
        model = MaskablePPO(
            "CnnPolicy",
            env,
            policy_kwargs=policy_kwargs,
            device="cuda",
            verbose=1,
            n_steps=N_STEPS,
            batch_size=N_STEPS // 2,
            n_epochs=4,
            gamma=0.95,
            ent_coef=0.01,
            learning_rate=lr_schedule,
            clip_range=clip_range_schedule,
            tensorboard_log=LOG_DIR
        )

    # Set the save directory
    if torch.backends.mps.is_available():
        save_dir = "trained_models_cnn_mps"
    else:
        save_dir = f"PPO_Snake_Game_{BOARD_SIZE}x{BOARD_SIZE}_CNN_v{VER_NUM}_{start_time}"
    os.makedirs(save_dir, exist_ok=True)

    # Callbacks
    # 1. Checkpoint Callback: Save model periodically
    checkpoint_interval = 15625 # checkpoint_interval * num_envs = total_steps_per_checkpoint
    checkpoint_callback = CheckpointCallback(save_freq=checkpoint_interval, save_path=save_dir, name_prefix=f"PPO_Snake_Game_{BOARD_SIZE}x{BOARD_SIZE}_CNN")

    # 2. Keyboard Stop Callback: Stop training safely by pressing 'q'
    stop_train_callback = KeyboardStopCallback(key='q')

    # Setup Logging
    original_stdout = sys.stdout
    log_file_path = os.path.join(save_dir, "training_log.txt")

    print(f"Starting training... Logs will be saved to: {log_file_path}")
    print("========================================")
    print("   Press and hold 'q' to stop training   ")
    print("========================================")

    # Switch stdout to DualLogger
    sys.stdout = DualLogger(log_file_path)

    try:
        model.learn(
            total_timesteps=int(100000000),
            # Add both callbacks to the list
            callback=[checkpoint_callback, stop_train_callback],
            tb_log_name=f"PPO_Snake_Game_{BOARD_SIZE}x{BOARD_SIZE}_CNN_v{VER_NUM}_{start_time}"
        )

        # Save the final model (Executed if training finishes naturally or is stopped by callback)
        final_model_path = os.path.join(save_dir, f"PPO_Snake_Game_{BOARD_SIZE}x{BOARD_SIZE}_CNN_final_v{VER_NUM}_{start_time}.zip")
        model.save(final_model_path)
        print(f"Training finished. Final model saved to {final_model_path}")

    except KeyboardInterrupt:
        # Fallback for Ctrl+C (though 'q' is preferred for stability)
        print("\nTraining interrupted by user (Ctrl+C). Saving current model...")

    except Exception as e:
        print(f"\nAn error occurred during training: {e}")
        raise

    finally:
        # Restore stdout and close the log file
        if hasattr(sys.stdout, 'close'):
            sys.stdout.close()
        sys.stdout = original_stdout

        # Safely close the environment
        try:
            env.close()
        except:
            pass
        print("Environment closed and stdout restored.")

if __name__ == "__main__":
    main()
