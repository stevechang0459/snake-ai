import os
import numpy as np
import torch as th
import onnxruntime as ort
from sb3_contrib import MaskablePPO

if th.backends.mps.is_available():
    MODEL_PATH = r"trained_models_cnn_mps/ppo_snake_final"
else:
    MODEL_PATH = r"trained_models_cnn_v3_20260217_012656/ppo_snake_final_v3_20260217_012656"

ONNX_PATH = "ppo_snake_cnn.onnx"

# We must redefine this class to ensure PyTorch inference logic matches the export exactly.
class OnnxableMaskablePolicy(th.nn.Module):
    def __init__(self, features_extractor, mlp_extractor, action_net):
        super().__init__()
        self.features_extractor = features_extractor
        self.mlp_extractor = mlp_extractor
        self.action_net = action_net

    def forward(self, obs):
        # Image Normalization (Must match export logic)
        preprocessed_obs = obs.float() / 255.0
        features = self.features_extractor(preprocessed_obs)
        latent_pi, _ = self.mlp_extractor(features)
        logits = self.action_net(latent_pi)
        return logits

def check_accuracy():
    print(f"Loading PyTorch model: {MODEL_PATH}...")
    # Force CPU usage to minimize floating-point errors between GPU and CPU.
    model = MaskablePPO.load(MODEL_PATH, device="cpu")

    # Wrap the model
    pytorch_model = OnnxableMaskablePolicy(
        model.policy.features_extractor,
        model.policy.mlp_extractor,
        model.policy.action_net
    )

    # Switch to eval mode. Otherwise, randomness from Dropout/BatchNorm will cause errors.
    pytorch_model.eval()

    print(f"Loading ONNX model: {ONNX_PATH}...")
    ort_session = ort.InferenceSession(ONNX_PATH)

    # 1. Prepare Test Data
    # Generate random input (Batch Size=1, Channel=3, H=84, W=84)
    # Simulate 0-255 image data
    dummy_input = th.randint(0, 256, (1, 3, 84, 84), dtype=th.float32)

    # Convert to numpy for ONNX usage
    onnx_input = {ort_session.get_inputs()[0].name: dummy_input.numpy()}

    # 2. Run PyTorch Inference
    with th.no_grad():
        torch_output = pytorch_model(dummy_input).numpy()

    # 3. Run ONNX Inference
    onnx_output = ort_session.run(None, onnx_input)[0]

    # 4. Calculate Error
    # MSE (Mean Squared Error)
    mse = np.mean((torch_output - onnx_output) ** 2)
    # Max Absolute Difference
    max_diff = np.max(np.abs(torch_output - onnx_output))

    print("\n" + "="*30)
    print("       Validation Report")
    print("="*30)
    print(f"PyTorch Output Sample:\n{torch_output[0]}")
    print(f"ONNX    Output Sample:\n{onnx_output[0]}")
    print("-" * 30)
    print(f"MSE (Mean Squared Error):       {mse:.10f}")
    print(f"Max Diff (Maximum Error):  {max_diff:.10f}")

    # 5. Verdict
    # Tiny floating-point errors (1e-5 ~ 1e-6) between frameworks are normal.
    # If the error is less than 1e-4, the conversion is generally considered successful.
    limit = 1e-4
    if max_diff < limit:
        print(f"\n[PASS] Success! Model conversion verified. Error is negligible (< {limit}).")
    else:
        print(f"\n[WARNING] Error seems high (> {limit}). Check if .eval() was missed or if preprocessing logic differs.")

if __name__ == "__main__":
    if os.path.exists(ONNX_PATH):
        check_accuracy()
    else:
        print(f"ONNX file not found: {ONNX_PATH}. Please run export_to_onnx.py first.")
