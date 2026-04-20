import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import scipy.sparse as sp
import torch

from model import models
from utility import calc_gso, calc_chebynet_gso


APP_DIR = Path(__file__).resolve().parent
ART_DIR = APP_DIR / "artifacts"
DATA_DIR = ART_DIR / "data_v2"
MODEL_DIR = APP_DIR / "models"


def safe_read_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_model_config(device: torch.device):
    args = SimpleNamespace(
        Kt=3,
        Ks=3,
        n_his=180,
        stblock_num=2,
        act_func="glu",
        graph_conv_type="cheb_graph_conv",
        gso_type="sym_norm_lap",
        enable_bias=True,
        droprate=0.3,
    )

    adj = sp.load_npz(DATA_DIR / "adj.npz").tocsc()
    gso = calc_gso(adj, args.gso_type)
    gso = calc_chebynet_gso(gso)
    gso_np = gso.toarray().astype(np.float32)
    args.gso = torch.from_numpy(gso_np).to(device)

    blocks = [
        [5],
        [32, 8, 32],
        [32, 8, 32],
        [32, 32],
        [5],
    ]
    n_vertex = 1505
    return args, blocks, n_vertex


def load_stgcn_model(device: torch.device):
    args, blocks, n_vertex = build_model_config(device)

    model = models.STGCNChebGraphConv(args, blocks, n_vertex).to(device)
    ckpt_path = MODEL_DIR / "stgcn_best.pt"
    state = torch.load(ckpt_path, map_location=device)

    if isinstance(state, dict) and "state_dict" in state:
        model.load_state_dict(state["state_dict"], strict=False)
    else:
        model.load_state_dict(state, strict=False)

    model.eval()
    return model


def make_dummy_input():
    meta = safe_read_json(DATA_DIR / "meta.json")
    lookback = int(meta.get("lookback", 180))
    n_types = int(meta.get("n_types", 5))
    n_grids = int(meta.get("n_grids", 1505))

    # 和你当前 app 的输入格式一致: (B, C, L, N)
    x = np.random.rand(1, n_types, lookback, n_grids).astype(np.float32)
    return torch.from_numpy(x)


def main():
    device = torch.device("cpu")
    model = load_stgcn_model(device)
    dummy_input = make_dummy_input()

    out_dir = MODEL_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = out_dir / "stgcn_best.onnx"

    # 先试新版导出
    try:
        torch.onnx.export(
            model,
            (dummy_input,),
            str(onnx_path),
            input_names=["x"],
            output_names=["pred"],
            opset_version=18,
            dynamo=True,
        )
        print(f"ONNX exported with dynamo=True -> {onnx_path}")
    except Exception as e:
        print(f"[WARN] dynamo=True export failed: {e}")
        print("Retrying with legacy exporter...")

        torch.onnx.export(
            model,
            (dummy_input,),
            str(onnx_path),
            input_names=["x"],
            output_names=["pred"],
            opset_version=17,
            dynamic_axes={
                "x": {0: "batch"},
                "pred": {0: "batch"},
            },
        )
        print(f"ONNX exported with legacy exporter -> {onnx_path}")

    # 简单验证一次输出形状
    with torch.no_grad():
        y = model(dummy_input)
    print("PyTorch output shape:", tuple(y.shape))


if __name__ == "__main__":
    main()
