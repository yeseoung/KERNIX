"""학습한 모델을 KERNIX .knx로 변환.

  python tools/convert.py model.pt  model.knx [--quantize int8] [--input-shape 3,32,32]   PyTorch (torch 필요)
  python tools/convert.py model.h5  model.knx [--quantize int8]                           Keras (tensorflow 필요)

PyTorch: torch.save(model, path)로 모델 전체를 저장한 파일. 일렬 MLP가 아니면 표본 하나의 입력 모양
(PyTorch 순서, 배치 제외)이 필요하다 (첫 층이 Linear면 자동). torch.load는 pickle이라 직접 만든 파일만 변환할 것.
Keras: .h5 또는 .keras. 입력 모양은 모델에서 읽는다.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def main(argv=None):
    ap = argparse.ArgumentParser(description="학습한 모델(PyTorch .pt / Keras .h5)을 KERNIX .knx로 변환")
    ap.add_argument("src", help="PyTorch .pt/.pth 또는 Keras .h5/.keras")
    ap.add_argument("dst", help="출력 .knx")
    ap.add_argument("--quantize", choices=["int8"], help="가중치를 int8로 (파일 약 1/3.5, 표본이 적을 때 더 빠름)")
    ap.add_argument("--input-shape", help="PyTorch: 표본 하나의 입력 모양 (배치 제외). 예: 3,32,32 또는 20,8")
    a = ap.parse_args(argv)
    ext = os.path.splitext(a.src)[1].lower()
    if ext in (".h5", ".keras"):
        if a.input_shape:
            ap.error("--input-shape는 PyTorch 모델에만 씀 (Keras는 모델에 입력 모양이 있음)")
        import tensorflow as tf
        from export_keras import export
        export(tf.keras.models.load_model(a.src, compile=False), a.dst, a.quantize)
        print(f"{a.src} → {a.dst} ({os.path.getsize(a.dst) / 1024:.1f} KB)")
    elif ext in (".pt", ".pth"):
        from export_torch import convert
        convert(a.src, a.dst, a.quantize, tuple(int(d) for d in a.input_shape.split(",")) if a.input_shape else None)
    else:
        ap.error(f"확장자 {ext}: .pt/.pth(PyTorch) 또는 .h5/.keras(Keras)만 지원")


if __name__ == "__main__":
    main()
