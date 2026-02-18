import onnx
import onnxruntime as ort
import numpy as np

if __name__ == "__main__":
    onnx_path = "ppo_snake_cnn.onnx"

    # 1. 檢查模型結構是否完整
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    print("ONNX 模型結構檢查通過！")

    # 2. 檢查 Input/Output 形狀
    session = ort.InferenceSession(onnx_path)
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    print(f"\nInput Name: {input_name}")
    print(f"Input Shape: {session.get_inputs()[0].shape}")
    # 預期輸出: ['batch_size', 3, 84, 84]

    print(f"\nOutput Name: {output_name}")
    print(f"Output Shape: {session.get_outputs()[0].shape}")
    # 預期輸出: ['batch_size', 4] (假設有 4 個動作)

    # 3. 試跑一次推論 (Sanity Check)
    # 模擬一張 84x84 的 RGB 圖片 (數值 0-255)
    dummy_input = np.random.randint(0, 256, (1, 3, 84, 84), dtype=np.uint8).astype(np.float32)
    result = session.run([output_name], {input_name: dummy_input})

    print(f"\n推論測試成功！")
    print(f"輸出 Logits: {result[0]}")
