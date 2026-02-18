import openvino.runtime as ov
import onnxruntime as ort
import numpy as np
import os

# Path Settings
ONNX_PATH = "ppo_snake_cnn.onnx"
# Prioritize the b1 version since we only need a single image (batch=1) for accuracy comparison
IR_PATH = "model_ir/snake_fp16.xml"
if not os.path.exists(IR_PATH):
    IR_PATH = "model_ir/snake_fp16_b1.xml"

def check_accuracy():
    if not os.path.exists(ONNX_PATH) or not os.path.exists(IR_PATH):
        print(f"Error: Model files not found.\nONNX: {ONNX_PATH}\nIR: {IR_PATH}")
        return

    print(f"Comparing Models:\n1. ONNX (FP32): {ONNX_PATH}\n2. IR (FP16):   {IR_PATH}")

    # 1. Prepare Test Data (Batch=1, Channel=3, Height=84, Width=84)
    # Simulate random image data (0-255)
    input_shape = (1, 3, 84, 84)
    # Note: Must convert to float32 as the model input layer typically expects float
    dummy_input = np.random.randint(0, 256, input_shape).astype(np.float32)

    # 2. Execute ONNX Inference (Baseline)
    print("Running ONNX inference...")
    ort_sess = ort.InferenceSession(ONNX_PATH)
    input_name_onnx = ort_sess.get_inputs()[0].name
    # ONNX Output
    onnx_results = ort_sess.run(None, {input_name_onnx: dummy_input})
    onnx_logits = onnx_results[0]

    # 3. Execute OpenVINO IR Inference (Test Value)
    print("Running OpenVINO IR inference...")
    core = ov.Core()
    # Load and compile model (Use CPU for validation to ensure a clean environment; NPU can also be used)
    compiled_model = core.compile_model(IR_PATH, "CPU")
    infer_request = compiled_model.create_infer_request()

    # OpenVINO Input
    # Get input node (usually index=0)
    input_node = compiled_model.input(0)
    results = infer_request.infer({input_node: dummy_input})

    # Get output node
    output_node = compiled_model.output(0)
    ir_logits = results[output_node]

    # 4. Calculate Error
    # Ensure dimensions match
    if onnx_logits.shape != ir_logits.shape:
        print(f"Warning: Output dimensions do not match! ONNX: {onnx_logits.shape}, IR: {ir_logits.shape}")
        return

    # Calculate MSE (Mean Squared Error)
    mse = np.mean((onnx_logits - ir_logits) ** 2)
    # Calculate Maximum Absolute Difference
    max_diff = np.max(np.abs(onnx_logits - ir_logits))

    print("\n" + "="*40)
    print("      Accuracy Validation Report (FP32 vs FP16)")
    print("="*40)
    print(f"ONNX (FP32) Output Sample: {onnx_logits[0]}")
    print(f"IR   (FP16) Output Sample: {ir_logits[0]}")
    print("-" * 40)
    print(f"MSE (Mean Squared Error):      {mse:.10f}")
    print(f"Max Diff (Maximum Error): {max_diff:.10f}")

    # Verdict Thresholds
    # FP16 quantization usually introduces errors in the range of 1e-3 ~ 1e-2, which is acceptable for RL.
    threshold = 0.05      # Loose threshold
    strict_threshold = 0.005 # Strict threshold

    if max_diff < strict_threshold:
        print(f"\n[PERFECT] Perfect! Error is negligible ({max_diff:.6f} < {strict_threshold})")
    elif max_diff < threshold:
        print(f"\n[PASS] Passed! Error is within acceptable range ({max_diff:.6f} < {threshold})")
        print("       (Slight precision loss from FP16 is normal)")
    else:
        print(f"\n[WARNING] Warning: High error detected ({max_diff:.6f} > {threshold})")
        print("       Please check the conversion process or if the model is highly sensitive to precision.")

    print("="*40)

if __name__ == "__main__":
    check_accuracy()
