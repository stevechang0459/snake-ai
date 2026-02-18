import openvino as ov
import os

ONNX_FILE = "ppo_snake_cnn.onnx"
OUTPUT_DIR = "model_ir"
MODEL_NAME = "snake_fp16"

# Set the batch size for the static shape
BATCH_SIZE = 8

def main():
    if not os.path.exists(ONNX_FILE):
        print(f"Error: {ONNX_FILE} does not exist.")
        return

    # Create output directory if it doesn't exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Initialize OpenVINO Core and read the ONNX model
    core = ov.Core()
    model = core.read_model(ONNX_FILE)

    # Define static shape: [Batch_Size, Channels, Height, Width]
    # Reshaping to a static batch size allows the NPU/GPU to optimize execution better than dynamic shapes.
    static_shape = [BATCH_SIZE, 3, 84, 84]
    model.reshape(static_shape)

    # Define output file path
    output_xml = os.path.join(OUTPUT_DIR, f"{MODEL_NAME}_b{BATCH_SIZE}.xml")

    # Save the model to OpenVINO IR format (.xml and .bin)
    # compress_to_fp16=True reduces model size by half with minimal accuracy loss, ideal for NPU inference.
    ov.save_model(
        model,
        output_model=output_xml,
        compress_to_fp16=True
    )

    print(f"Output files: {output_xml}, {output_xml.replace('.xml', '.bin')}")
    print(f"Successfully converted ONNX Model to OpenVINO IR format.")

if __name__ == "__main__":
    main()
