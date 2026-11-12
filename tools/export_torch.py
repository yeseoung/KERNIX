"""PyTorch 모델 → .knx. 일렬 MLP(nn.Sequential)는 KNX1/KNX2(export), 그 밖은 torch.fx 그래프로 KNX3(torch_graph.py).
명령줄은 tools/convert.py"""
import os
import struct

import torch
import torch.nn as nn

from knx_quant import MAGIC, weight_block

ACTS = {nn.ReLU: 0, nn.GELU: 1, nn.SiLU: 2, nn.Sigmoid: 3, nn.Tanh: 4}


def floats(t):
    return t.detach().contiguous().float().numpy().astype("<f4").tobytes()


def export(model, path, quantize=None):
    """형식: "KNX1", uint32 n_ops, 연산 목록 (docs/mlp-runtime-design.md).
    quantize="int8"이면 "KNX2"로 쓰고 Linear 가중치를 I8_COL(출력 채널별 int8)로 (docs/int8-quantization-design.md)"""
    v2 = quantize is not None
    if quantize not in MAGIC:
        raise ValueError(f"지원하지 않는 quantize: {quantize}")
    ops = []
    for layer in model:
        if isinstance(layer, nn.Linear):
            # KERNIX는 x @ W를 바로 계산하도록 in×out 방향으로 저장 (torch는 out×in)
            op = struct.pack("<4I", 0, layer.in_features, layer.out_features, layer.bias is not None)
            op += weight_block(layer.weight.detach().T.float().numpy(), "i8_col" if v2 else "f32", v2)
            if layer.bias is not None:
                op += floats(layer.bias)
        elif type(layer) in ACTS:
            if isinstance(layer, nn.GELU) and layer.approximate != "none":
                raise ValueError("GELU는 approximate='none'만 지원")
            op = struct.pack("<2I", 1, ACTS[type(layer)])
        elif isinstance(layer, nn.LayerNorm):
            if len(layer.normalized_shape) != 1:
                raise ValueError("LayerNorm은 마지막 차원 하나만 지원")
            cols = layer.normalized_shape[0]
            op = struct.pack("<3If", 2, cols, layer.weight is not None, layer.eps)
            if layer.weight is not None:
                op += floats(layer.weight)
                op += floats(layer.bias if layer.bias is not None else torch.zeros(cols))
        elif isinstance(layer, nn.Softmax):
            if layer.dim not in (-1, 1):
                raise ValueError("Softmax는 마지막 차원(dim=-1)만 지원")
            op = struct.pack("<I", 3)
        else:
            raise ValueError(f"지원하지 않는 레이어: {type(layer).__name__}")
        ops.append(op)
    with open(path, "wb") as f:
        f.write(MAGIC[quantize] + struct.pack("<I", len(ops)) + b"".join(ops))


def convert(src, dst, quantize=None, input_shape=None):
    """torch.save(model, path)로 저장한 모델을 .knx로. 일렬 MLP(nn.Sequential)는 KNX1/KNX2, 그 밖(합성곱, RNN,
    Transformer, 잔차 연결 등)은 torch.fx로 그래프를 뽑아 KNX3 (torch_graph.py, 입력 모양 필요)"""
    # torch.save(model)은 pickle이라 weights_only=False가 필요: 파일 안의 코드가 실행되므로 직접 만든 파일만 변환할 것
    model = torch.load(src, map_location="cpu", weights_only=False)
    if not isinstance(model, nn.Module):
        raise ValueError(f"nn.Module이 아님 ({type(model).__name__}). state_dict가 아니라 torch.save(model, path)로 저장할 것")
    model.eval()
    try:
        if not isinstance(model, nn.Sequential):
            raise ValueError("nn.Sequential이 아님")
        export(model, dst, quantize)
    except ValueError as seq_err:
        from torch_graph import export_graph
        if input_shape is None:
            first = next((m for m in model.modules() if not isinstance(m, (nn.Sequential, type(model)))), None)
            if not isinstance(first, nn.Linear):
                raise ValueError(f"입력 모양을 알 수 없음: --input-shape로 표본 하나의 모양을 지정 (예: 3,32,32). ({seq_err})") from None
            input_shape = (first.in_features,)
        layout = export_graph(model, dst, input_shape, quantize)
        if layout:
            print("입력은 PyTorch처럼 channels-first (batch, C, ...)로 주면 Python 바인딩이 축을 옮김")
    print(f"{src} → {dst} ({os.path.getsize(dst) / 1024:.1f} KB)")
