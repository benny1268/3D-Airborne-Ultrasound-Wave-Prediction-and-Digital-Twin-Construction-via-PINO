import torch
import time
from physicsnemo.models.fno import FNO
from physicsnemo.distributed import DistributedManager
from U_FNO import U_FNO
from F_FNO import F_FNO

# 初始化分佈式管理器
DistributedManager.initialize()
dist = DistributedManager()

if torch.cuda.is_available():
    print("CUDA is available. Running on GPU.")
else:
    print("CUDA is not available. Running on CPU.")

# 測試不同的輸出通道數（你可以改成 num_fno_modes 也可）
out_channels_list = [1, 5, 7, 10, 15, 20]
num_runs = 1
total_steps = 200

# 假輸入資料：batch=3，4通道輸入，大小為 128x128x256
dummy_input = torch.randn(1, 5, 150, 150, 300).to(dist.device)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

for i, out_channels in enumerate(out_channels_list):
    runtimes = []
    mem_usages = []
    print(f"\n=== Testing Out Channels = {out_channels} ===")

    for run in range(num_runs):

        # 清空記憶體統計
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.backends.cudnn.benchmark = True

        # 建立 FNO 模型
        model = F_FNO(
            in_channels=5,
            out_channels=10,
            decoder_layers=1,
            decoder_layer_size=128,
            dimension=3,
            latent_channels=32,
            num_fno_layers=8,
            num_fno_modes=50,
            padding=0,
        ).to(dist.device)

        # 顯示參數量
        total_params = count_parameters(model) / 1e6
        print(f"Model Parameters: {total_params:.2f} M parameters")

        # 計時推論
        torch.cuda.synchronize()
        start_time = time.time()
        with torch.no_grad():
            for _ in range(0, total_steps, 10):
                for j in range(1):
                    start_time_one = time.time()
                    _ = model(dummy_input)
                    end_time_one = time.time()
                    print(f"Inference step time: {end_time_one - start_time_one:.5f} sec")
        torch.cuda.synchronize()
        end_time = time.time()

        # 記錄記憶體使用量
        mem_used = torch.cuda.max_memory_allocated(device=dist.device) / 1024**3
        mem_usages.append(mem_used)
        runtimes.append(end_time - start_time)

        print(f"GPU Memory Usage: {mem_used:.2f} GB")
        print(f"Total Run Time: {end_time - start_time:.5f} sec")

        # 釋放模型
        del model
        torch.cuda.empty_cache()

    # 統計
    avg_time = sum(runtimes) / num_runs
    avg_mem = sum(mem_usages) / num_runs
    print(f"Average Inference Time : {avg_time:.5f} sec")
    print(f"Average Peak Memory Usage: {avg_mem:.2f} GB")
