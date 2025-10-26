import torch
import torch.nn as nn
import torch.nn.functional as F
from physicsnemo.models.fno import FNO
from physicsnemo.distributed import DistributedManager
from U_FNO import U_FNO
from F_FNO import F_FNO

# === 初始化分布式管理器 ===
DistributedManager.initialize()
dist = DistributedManager()

# === 裝置與 AMP 支援狀態 ===
device = dist.device
use_amp = torch.cuda.is_available()
print(f"✅ Running on: {device}")


# === 模型建立 ===
model = FNO(
    in_channels=5,
    out_channels=5,
    decoder_layers=1,
    decoder_layer_size=128,
    dimension=2,
    latent_channels=32,
    num_fno_layers=8,
    num_fno_modes=32,
    padding=0,
).to(device)
model.eval()

# === 測試輸入資料 ===
input_tensor = torch.randn(1, 5, 64, 64).to(device)

# === 前向推論測試 ===
with torch.no_grad():
    output = model(input_tensor)
print(f"✅ Output shape: {output.shape}")

# === 模型參數統計 ===
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"✅ Total parameters: {total_params:,}")
print(f"✅ Trainable parameters: {trainable_params:,}")

# === TorchScript 測試 ===
try:
    scripted_model = torch.jit.script(model)
    scripted_out = scripted_model(input_tensor)
    print(f"✅ TorchScript succeeded. Scripted output shape: {scripted_out.shape}")
except Exception as e:
    print(f"❌ TorchScript failed: {e}")

# === CUDA Graph 測試 ===
if device.type == "cuda":
    static_input = input_tensor.clone().detach()
    static_input.requires_grad = False

    # 預熱 CUDA kernels
    _ = model(static_input)
    torch.cuda.synchronize()

    g = torch.cuda.CUDAGraph()
    model.zero_grad(set_to_none=True)

    with torch.cuda.graph(g):
        output = model(static_input)

    print("✅ CUDA Graph captured and executed successfully.")
else:
    print("⚠️ CUDA Graphs only available on CUDA devices.")
