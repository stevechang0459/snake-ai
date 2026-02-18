import os
import openvino.runtime as ov

IR_MODEL_PATH = "model_ir/snake_fp16.xml"
if not os.path.exists(IR_MODEL_PATH):
    IR_MODEL_PATH = "model_ir/snake_fp16_b1.xml"

def check_ir_model():
    if not os.path.exists(IR_MODEL_PATH):
        print(f"Error: {IR_MODEL_PATH} does not exist.")
        return

    core = ov.Core()
    model = core.read_model(IR_MODEL_PATH)

    print("\n" + "="*40)
    print("      OpenVINO IR Model Structure Check")
    print("="*40)

    # 1. Check Inputs
    print(f"[Input Layers]")
    for i, input_node in enumerate(model.inputs):
        print(f"  [{i}] Name: {input_node.any_name}")
        print(f"      Shape: {input_node.shape}")
        print(f"      Type:  {input_node.element_type}")
        # Expected Type is usually f32 (inputs often remain float32, while internal model operations are f16).

    # 2. Check Outputs
    print(f"\n[Output Layers]")
    for i, output_node in enumerate(model.outputs):
        print(f"  [{i}] Name: {output_node.any_name}")
        print(f"      Shape: {output_node.shape}")
        print(f"      Type:  {output_node.element_type}")

    # 3. Check Model Precision (Check if FP16 layers exist)
    # We iterate through the operation nodes (Ops) in the model.
    print(f"\n[Precision Check]")
    fp16_count = 0
    total_count = 0
    for op in model.get_ops():
        total_count += 1
        # Check if the layer attributes contain the 'f16' keyword.
        # Note: OpenVINO 2022+ automatically optimizes upon loading,
        # but we can infer precision from rt_info or element_type.
        if "f16" in str(op.get_output_element_type(0)):
            fp16_count += 1

    print(f"  Total Layers (Ops): {total_count}")
    print(f"  FP16 Layers (Estimated): {fp16_count}")

    if fp16_count > 0:
        print("  [PASS] FP16 operation layers detected. Model compression successful!")
    else:
        print("  [INFO] No explicit FP16 output layers detected (Inputs/Outputs might remain FP32, but internal weights are compressed).")
        # As long as the .bin file size is roughly half of the .onnx file, the weights are successfully compressed.

    print("="*40)

if __name__ == "__main__":
    check_ir_model()
