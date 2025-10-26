# Train

本資料夾為**PINO 模型訓練流程**，使用 NVIDIA PhysicsNeMo 套件，並以 `Trainingdata_generation/` 資料夾產生的聲場模擬資料進行學習。

## 環境建議

- 建議使用 `conda` 建立獨立環境，避免與 `jwave` 模擬環境混用。
- 建議使用具有 CUDA 支援的 GPU。

---

## 安裝步驟

### (1) 安裝核心套件：

```bash
pip install -r nv_physicsnemo_requirement.txt
```

### (2) 安裝 PhysicsNeMo-Sym 模組

```bash
pip install nvidia-physicsnemo-sym --no-build-isolation
```

如遇錯誤，請參考官方教學： [https://docs.nvidia.com/deeplearning/physicsnemo/getting-started/index.html](https://docs.nvidia.com/deeplearning/physicsnemo/getting-started/index.html)

---

## 驗證安裝是否成功

```bash
python physicsnemo_check.py
```

若輸出 `torch.Size([128, 64])` 即表示安裝成功。

---

## 執行訓練

主訓練腳本為：

```bash
python train.py
```

可修改的訓練參數集中於：

```bash
conf/config_pino.yaml
```

例如：

- 訓練輪數（epochs）
- 模型結構（FNO 層數、channel 數量）
- 批次大小、學習率

---

## 詳細教學

https://hackmd.io/@benny88129/Bk7fCRjRge

