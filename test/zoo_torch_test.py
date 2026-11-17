"""PyTorch 모델 모음: 모듈·연산 종류마다 작은 모델을 무작위 가중치로 만들어 .knx로 내보내고, KERNIX 출력이 PyTorch와 같은지 본다.
  make test-zoo   (먼저 make python-lib)
"""
import os
import sys
import tempfile

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "tools"))
sys.path.insert(0, os.path.join(HERE, "..", "python"))
from export_torch import convert  # noqa: E402
from kernix import Model  # noqa: E402

torch.manual_seed(0)
fail = 0


def randomize(model):
    """BatchNorm 통계를 기본값(0, 1)이 아닌 값으로: 접어 넣기·채널 순서 오류가 드러나게"""
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
            m.running_mean.uniform_(-0.5, 0.5)
            m.running_var.uniform_(0.5, 2.0)
            m.weight.data.uniform_(0.5, 1.5)
            m.bias.data.uniform_(-0.5, 0.5)


def check(name, model, x, quantize=None, tol=2e-4):
    global fail
    randomize(model)
    model.eval()
    with torch.no_grad():
        want = model(x).numpy()
    with tempfile.TemporaryDirectory() as d:
        src, dst = os.path.join(d, "m.pt"), os.path.join(d, "m.knx")
        torch.save(model, src)
        try:
            import contextlib
            import io
            with contextlib.redirect_stdout(io.StringIO()):
                convert(src, dst, quantize, tuple(x.shape[1:]) if not isinstance(model, nn.Sequential) or x.dim() > 2 else None)
            got = Model(dst)(x.numpy())
            fmt = open(dst, "rb").read(4).decode()
        except Exception as e:  # noqa: BLE001  내보내기·실행 실패도 결과로 보고
            print(f"  ✗ {name}: {type(e).__name__}: {e}")
            fail += 1
            return
    err = np.abs(got - want).max() / max(1.0, np.abs(want).max()) if got.shape == want.shape else float("inf")
    ok = err <= tol
    print(f"  {'✓' if ok else '✗'} {name} [{fmt}{' int8' if quantize else ''}]: 출력 {want.shape[1:]}, 상대 오차 {err:.1e}"
          + ("" if got.shape == want.shape else f" (모양 {got.shape} != {want.shape})"))
    fail += not ok


class MeanLast(nn.Module):
    def forward(self, x):
        return x.mean(dim=-1)


class Block(nn.Module):  # ResNet 기본 블록 + SE
    def __init__(self, c):
        super().__init__()
        self.c1, self.b1 = nn.Conv2d(c, c, 3, padding=1, bias=False), nn.BatchNorm2d(c)
        self.c2, self.b2 = nn.Conv2d(c, c, 3, padding=1, bias=False), nn.BatchNorm2d(c)
        self.se1, self.se2 = nn.Linear(c, c // 2), nn.Linear(c // 2, c)

    def forward(self, x):
        h = F.relu(self.b1(self.c1(x)))
        h = self.b2(self.c2(h))
        s = torch.sigmoid(self.se2(F.relu(self.se1(h.mean(dim=(2, 3))))))
        return F.relu(h * s.view(s.size(0), -1, 1, 1) + x)


class ResNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(3, 8, 3, stride=2, padding=1), nn.BatchNorm2d(8), nn.ReLU())
        self.blocks = nn.Sequential(Block(8), Block(8))
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(8, 5)

    def forward(self, x):
        return self.fc(torch.flatten(self.pool(self.blocks(self.stem(x))), 1))


class Mixer(nn.Module):  # depthwise, Upsample, 패딩, cat(dim=1)
    def __init__(self):
        super().__init__()
        self.dw = nn.Conv2d(4, 8, 3, groups=4, padding=1)
        self.pw = nn.Conv2d(8, 6, 1)
        self.up = nn.Upsample(scale_factor=2)
        self.pad = nn.ZeroPad2d((1, 0, 2, 1))

    def forward(self, x):
        h = F.hardswish(self.pw(self.dw(x)))
        h = torch.cat([self.pad(self.up(h)), F.pad(self.up(h), (0, 1, 1, 2))], dim=1)
        return h.flatten(1)


class SeqConv(nn.Module):  # (B, T, F) → transpose → Conv1d → 다시 transpose → Linear
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv1d(5, 7, 3, padding="same", dilation=2)
        self.pool = nn.MaxPool1d(2)
        self.fc = nn.Linear(7, 3)

    def forward(self, x):
        h = self.pool(F.gelu(self.conv(x.transpose(1, 2))))
        return self.fc(h.permute(0, 2, 1)).mean(dim=1)


class LSTMHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(6, 8, num_layers=2, batch_first=True)
        self.fc = nn.Linear(8, 3)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])


class BiGRU(nn.Module):
    def __init__(self):
        super().__init__()
        self.gru = nn.GRU(4, 5, batch_first=True, bidirectional=True)
        self.rnn = nn.RNN(10, 4, batch_first=True, nonlinearity="relu")
        self.fc = nn.Linear(10 + 4, 2)

    def forward(self, x):
        out, h = self.gru(x)
        r, _ = self.rnn(out)
        return self.fc(torch.cat([h[-2], h[-1], r[:, 0]], dim=1))


class LSTMhn(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(3, 6, batch_first=True)

    def forward(self, x):
        _, (h, _) = self.lstm(x)
        return h[-1]


class GPT(nn.Module):  # 토큰 임베딩 + 위치 임베딩 + 인과 Transformer
    def __init__(self, norm_first, act):
        super().__init__()
        self.tok = nn.Embedding(30, 16)
        self.pos = nn.Parameter(torch.randn(1, 10, 16) * 0.1)
        layer = nn.TransformerEncoderLayer(16, 4, 32, dropout=0.0, activation=act, batch_first=True, norm_first=norm_first)
        self.enc = nn.TransformerEncoder(layer, 2, norm=nn.LayerNorm(16), enable_nested_tensor=False)
        self.head = nn.Linear(16, 30)
        self.register_buffer("mask", nn.Transformer.generate_square_subsequent_mask(10))

    def forward(self, ids):
        h = self.tok(ids) + self.pos
        return self.head(self.enc(h, mask=self.mask, is_causal=True))[:, -1]


class CrossAttn(nn.Module):
    def __init__(self):
        super().__init__()
        self.q, self.kv = nn.Linear(6, 12), nn.Linear(6, 8)
        self.attn = nn.MultiheadAttention(12, 3, batch_first=True, kdim=8, vdim=8)
        self.ln = nn.LayerNorm(12)

    def forward(self, x):
        q, kv = self.q(x), self.kv(x)
        a, _ = self.attn(q, kv, kv, need_weights=False)
        return self.ln(q + a)[:, 0]


def main():
    acts = [nn.Hardswish(), nn.Hardsigmoid(), nn.SiLU(), nn.GELU("tanh"), nn.Mish(), nn.ELU(0.5), nn.SELU(), nn.Softplus(),
            nn.ReLU6(), nn.LeakyReLU(0.05), nn.Softsign(), nn.Tanh(), nn.Sigmoid(), nn.PReLU(6)]

    print("=== PyTorch 모델 모음 → KERNIX ===")
    check("MLP 기존 형식 (nn.Sequential)", nn.Sequential(nn.Linear(10, 16), nn.ReLU(), nn.Linear(16, 4), nn.Softmax(dim=-1)), torch.randn(5, 10))
    check("MLP + BatchNorm1d + Dropout + PReLU", nn.Sequential(nn.Linear(10, 16), nn.BatchNorm1d(16), nn.PReLU(), nn.Dropout(0.3), nn.Linear(16, 3)),
          torch.randn(5, 10))
    check("활성화 14종", nn.Sequential(*[m for a in acts for m in (nn.Linear(6, 6), a)]), torch.randn(5, 6) * 0.5)
    check("CNN (BN, MaxPool, groups, AvgPool, AdaptiveAvgPool)", nn.Sequential(
        nn.Conv2d(3, 8, 3, padding=1), nn.BatchNorm2d(8), nn.ReLU(), nn.MaxPool2d(2),
        nn.Conv2d(8, 12, 3, stride=2, groups=2), nn.AvgPool2d(2, padding=1), nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(12, 4)),
        torch.randn(4, 3, 17, 15))
    check("CNN NCHW flatten (GAP 없이)", nn.Sequential(nn.Conv2d(2, 4, (3, 2), stride=(1, 2)), nn.ReLU(), nn.Flatten(), nn.Linear(4 * 6 * 3, 3)),
          torch.randn(3, 2, 8, 6))
    check("CNN 특징 맵 출력 (channels-first 그대로)", nn.Sequential(nn.Conv2d(3, 5, 3, padding=1), nn.Softmax(dim=1)), torch.randn(2, 3, 6, 7))
    check("Conv1d (C, L) + MaxPool1d + mean", nn.Sequential(nn.Conv1d(4, 6, 5, padding=2), nn.ReLU(), nn.MaxPool1d(2),
          MeanLast()), torch.randn(4, 4, 21))
    check("ResNet 블록 2개 + SE (잔차, view, mean)", ResNet(), torch.randn(3, 3, 16, 16))
    check("Depthwise · Upsample · ZeroPad2d · F.pad · cat(dim=1)", Mixer(), torch.randn(2, 4, 5, 6))
    check("시퀀스 Conv1d (transpose/permute)", SeqConv(), torch.randn(3, 12, 5))
    check("LSTM 2층 → out[:, -1]", LSTMHead(), torch.randn(4, 7, 6))
    check("양방향 GRU h_n[-2], h_n[-1] + RNN(relu)", BiGRU(), torch.randn(3, 6, 4))
    check("LSTM (h_n, c_n) → h_n[-1]", LSTMhn(), torch.randn(3, 5, 3))
    for nf, act in ((False, "relu"), (True, "gelu")):
        check(f"GPT (Embedding, 위치 파라미터, 인과 TransformerEncoder norm_first={nf} {act})", GPT(nf, act),
              torch.randint(0, 30, (3, 10)))
    check("교차 MultiheadAttention (kdim/vdim) + LayerNorm", CrossAttn(), torch.randn(3, 5, 6))
    check("CNN int8", nn.Sequential(nn.Conv2d(3, 8, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(8, 4)),
          torch.randn(4, 3, 8, 8), quantize="int8", tol=5e-2)

    print("통과 ✓" if fail == 0 else f"실패 {fail}개 ✗")
    sys.exit(fail != 0)


if __name__ == "__main__":
    main()
