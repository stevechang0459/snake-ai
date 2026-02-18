import torch as th
from sb3_contrib import MaskablePPO

# Determine the model path based on the available device
if th.backends.mps.is_available():
    MODEL_PATH = r"trained_models_cnn_mps/ppo_snake_final"
else:
    MODEL_PATH = r"trained_models_cnn_v3_20260217_012656/ppo_snake_final_v3_20260217_012656"

class OnnxableMaskablePolicy(th.nn.Module):
    def __init__(self, features_extractor, mlp_extractor, action_net):
        super().__init__()
        self.features_extractor = features_extractor
        self.mlp_extractor = mlp_extractor
        self.action_net = action_net
        # We don't need to calculate the Value since we are strictly doing inference, not training.
        # The Critic's job (Value Net) is to 'predict' how much reward we can get.
        # self.value_net = value_net

    def forward(self, obs):
        # 1. Image Normalization
        # Converts input from 0-255 range to 0.0-1.0 range
        preprocessed_obs = obs.float() / 255.0

        # 2. Feature Extraction (CNN)
        # Data flow: obs --> [features_extractor] --> features
        features = self.features_extractor(preprocessed_obs)

        # 3. Process through MLP Extractor
        # SB3's architecture splits this into latent_pi (policy) and latent_vf (value function)
        # Data flow: features --> [mlp_extractor] --> latent_pi (policy latent), latent_vf (value latent)
        latent_pi, _ = self.mlp_extractor(features)

        # # Calculate Value (Critic's job - Not needed for inference)
        # # Data flow: latent_vf --> [value_net] --> values
        # # values = self.value_net(latent_vf)

        # # Calculate Action Distribution (Actor's job - usually for training/stochastic sampling)
        # # Data flow: latent_pi --> [_get_action_dist_from_latent] --> distribution
        # # distribution = self._get_action_dist_from_latent(latent_pi)
        # # actions = distribution.get_actions(deterministic=deterministic)
        # # log_prob = distribution.log_prob(actions)
        # # actions = actions.reshape((-1, *self.action_space.shape))
        # # return actions, values, log_prob

        # 4. Output Action Logits (Raw scores before Softmax or Masking)
        # We return logits directly for ONNX so the inference engine can handle the final selection
        logits = self.action_net(latent_pi)

        return logits

def export_to_onnx():
    # Load the trained SnakeAI model, including weights and architecture
    print(f"Loading model from: {MODEL_PATH}")
    model = MaskablePPO.load(MODEL_PATH, device="cpu")
    output_path = "ppo_snake_cnn.onnx"

    # Create the exportable Policy object
    onnxable_model = OnnxableMaskablePolicy(
        model.policy.features_extractor,
        model.policy.mlp_extractor,
        model.policy.action_net
    )

    # Ensure the model is in eval mode
    # (This is crucial as it affects the behavior of layers like Dropout or BatchNorm)
    onnxable_model.eval()

    # Create Dummy Input (Used to trace the model structure)
    # According to snake_game_custom_wrapper_cnn.py, the observation space is (84, 84, 3)
    # However, SB3 internally uses Channel-First format (C, H, W)
    # Therefore, the input must be (Batch_Size, Channel, Height, Width)
    dummy_input = th.randn(1, 3, 84, 84)

    # Export to ONNX
    print(f"Exporting model to {output_path}...")
    th.onnx.export(
        onnxable_model,
        dummy_input,
        output_path,
        opset_version=14,
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes={
            "input": {0: "batch_size"},   # Allow variable batch size for input
            "logits": {0: "batch_size"}   # Allow variable batch size for output
        }
    )
    print(f"Success! Model exported to {output_path}")

if __name__ == "__main__":
    export_to_onnx()
