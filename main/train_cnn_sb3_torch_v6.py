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
from torch.utils.tensorboard import SummaryWriter

# ======================================
# Global Configuration
# ======================================

VER_NUM = "6"
LOAD_MODEL = False

if torch.backends.mps.is_available():
    NUM_ENV = 32 * 2
else:
    NUM_ENV = 32

BOARD_SIZE = 21
TARGET_STEPS = 100000000
N_STEPS = 2048
BATCH_SIZE = 2048
N_EPOCHS = 4
GAMMA = 0.995
ENT_COEF = 0.01
INITIAL_LR = 2.5e-4
FINAL_LR = 2.5e-6
INITIAL_CR = 0.150
FINAL_CR = 0.025

LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)

# ======================================
# Utility Classes
# ======================================

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

class TrainingSetupCallback(BaseCallback):
    def __init__(self, verbose=1):
        super().__init__(verbose)
        self.dual_logger = None
        self.original_stdout = sys.stdout

    def _on_training_start(self) -> None:
        if self.logger.dir is not None:
            # # Export Neural Network Graph to TensorBoard
            # try:
            #     writer = SummaryWriter(log_dir=self.logger.dir)
            #     # Construct a dummy observation matching the env output shape [Batch, Channels, Height, Width]
            #     dummy_obs = torch.as_tensor(self.model.observation_space.sample()).float().unsqueeze(0).to(self.model.device)
            #     # Trace the entire policy architecture
            #     writer.add_graph(self.model.policy, dummy_obs)
            #     writer.close()

            #     if self.verbose > 0:
            #         print("Successfully exported CustomCNN architecture to TensorBoard GRAPHS.")

            # except Exception as e:
            #     print(f"Error occurred while exporting architecture graph: {e}")

            # Setup Logging
            log_file_path = os.path.join(self.logger.dir, "training_log.txt")
            self.dual_logger = DualLogger(log_file_path)
            # Switch stdout to DualLogger
            sys.stdout = self.dual_logger

            if self.verbose > 0:
                print(f"Starting training... Logs will be saved to: {log_file_path}")
                print("Press and hold 'q' to stop training...")

    def _on_training_end(self) -> None:
        if self.dual_logger is not None:
            sys.stdout = self.original_stdout
            self.dual_logger.close()

    def _on_step(self) -> bool:
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

# ======================================
# Helper Functions
# ======================================

# Linear scheduler for learning rate and clip range
def linear_schedule(initial_value, final_value=0.0):
    if isinstance(initial_value, str):
        initial_value = float(initial_value)
        final_value = float(final_value)
        assert (initial_value > 0.0)

    def scheduler(progress):
        return final_value + progress * (initial_value - final_value)

    return scheduler

def get_resumed_rate(initial_lr, final_lr, total_goal_steps, current_steps):
    """ Calculates the correct starting LR based on global progress """
    progress = current_steps / total_goal_steps
    current_lr = initial_lr - (progress * (initial_lr - final_lr))
    return max(current_lr, final_lr)

def make_env(seed=0):
    def _init():
        # Note: board_size is set to 21 here as per your request
        env = SnakeEnv(seed=seed, board_size=BOARD_SIZE, limit_step=True)
        env = ActionMasker(env, SnakeEnv.get_action_mask)
        env = Monitor(env)
        env.seed(seed)
        return env
    return _init

# ======================================
# Main Training Loop
# ======================================

def main():
    start_time = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Generate a list of unique random seeds for each environment
    seed_set = set()
    while len(seed_set) < NUM_ENV:
        seed_set.add(random.randint(0, int(1e9)))

    # Create the vectorized Snake environment (Multi-process)
    env = SubprocVecEnv([make_env(seed=s) for s in seed_set])
    env = VecTransposeImage(env)

    policy_kwargs = dict(
        features_extractor_class=CustomCNN,
        features_extractor_kwargs=dict(features_dim=512),
        net_arch=dict(pi=[256, 256], vf=[256, 256])
    )

    if LOAD_MODEL:
        start_time = "20260220_162010"
        steps = 500000
        MODEL_PATH = rf"PPO_Snake_Game_21x21_CNN_v{VER_NUM}_{start_time}/PPO_Snake_Game_{BOARD_SIZE}x{BOARD_SIZE}_CNN_v{VER_NUM}_{steps}_steps"

        custom_lr_schedule = linear_schedule(INITIAL_LR, FINAL_LR)
        custom_clip_range_schedule = linear_schedule(INITIAL_CR, FINAL_CR)
        custom_objects = {
            "gamma": GAMMA,
            "ent_coef": ENT_COEF,
            "batch_size": N_STEPS,
            "learning_rate": custom_lr_schedule,
            "clip_range": custom_clip_range_schedule,
        }
        try:
            model = MaskablePPO.load(
                MODEL_PATH,
                env=env,
                tensorboard_log=LOG_DIR,
                custom_objects=custom_objects
            )
        except Exception as e:
            print(f"\nError occurred while loading model: {e}")
            return
    else:
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
            lr_schedule = linear_schedule(INITIAL_LR, FINAL_LR)
            clip_range_schedule = linear_schedule(INITIAL_CR, FINAL_CR)
            model = MaskablePPO(
                "CnnPolicy",
                env,
                policy_kwargs=policy_kwargs,
                device="cuda",
                verbose=1,
                n_steps=N_STEPS,
                batch_size=BATCH_SIZE,
                n_epochs=N_EPOCHS,
                gamma=GAMMA,
                ent_coef=ENT_COEF,
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
    # Checkpoint Callback: Save model periodically
    checkpoint_interval = 15625
    checkpoint_callback = CheckpointCallback(save_freq=checkpoint_interval, save_path=save_dir, name_prefix=f"PPO_Snake_Game_{BOARD_SIZE}x{BOARD_SIZE}_CNN_v{VER_NUM}")

    # Keyboard Stop Callback: Stop training safely by pressing 'q'
    stop_train_callback = KeyboardStopCallback(key='q')

    original_stdout = sys.stdout
    setup_callback = TrainingSetupCallback(verbose=1)

    # Execute Training
    try:
        if LOAD_MODEL:
            model.learn(
                total_timesteps=int(TARGET_STEPS - steps),
                reset_num_timesteps=False,
                callback=[setup_callback, checkpoint_callback, stop_train_callback],
                tb_log_name=f"PPO_Snake_Game_{BOARD_SIZE}x{BOARD_SIZE}_CNN_v{VER_NUM}_{start_time}"
            )
        else:
            model.learn(
                total_timesteps=int(TARGET_STEPS),
                callback=[setup_callback, checkpoint_callback, stop_train_callback],
                tb_log_name=f"PPO_Snake_Game_{BOARD_SIZE}x{BOARD_SIZE}_CNN_v{VER_NUM}_{start_time}"
            )

        # Save the final model (Executed if training finishes naturally or is stopped by callback)
        final_model_path = os.path.join(save_dir, f"PPO_Snake_Game_{BOARD_SIZE}x{BOARD_SIZE}_CNN_final_v{VER_NUM}_{start_time}.zip")
        model.save(final_model_path)
        print(f"Training finished. Final model saved to {final_model_path}")

    except KeyboardInterrupt:
        # Fallback for Ctrl+C (though 'q' is preferred for stability)
        print("Training interrupted by user (Ctrl+C). Saving current model...")

    except Exception as e:
        print(f"Error occurred during training: {e}")
        raise

    finally:
        # Restore stdout and close the log file
        if isinstance(sys.stdout, DualLogger):
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
