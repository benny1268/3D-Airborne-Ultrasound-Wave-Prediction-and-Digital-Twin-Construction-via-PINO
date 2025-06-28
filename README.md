# 🌀 3D Airborne Ultrasound Wave Prediction and Digital Twin Construction via Physics-Informed Neural Operators (PINO)

本專案提供一套完整流程，用於訓練**空氣超音波陣列物理模擬的 PINO 模型**，包含：

- 聲場模擬（使用 [`jwave`](https://github.com/ucl-bug/jwave)）
- AI 模型訓練（使用 [NVIDIA PhysicsNeMo](https://developer.nvidia.com/physicsnemo)）

---

## 專案結構說明

建議將本專案拆為兩個資料夾，並為每個資料夾建立獨立的 Conda 環境，以避免相依套件衝突。

### ✅ 子專案一：`Trainingdata_generation/`
聲場模擬資料生成，使用 `jwave` 套件（基於 `JAX` 的 CUDA 加速超音波模擬工具）

### ✅ 子專案二：`Train/`
基於 NVIDIA PhysicsNeMo 的 PINO 模型訓練腳本，透過 `jwave` 生成的資料進行 3D 聲場預測訓練。

---

### ► 詳細使用方式請見：
- [`Trainingdata_generation/README.md`](./Trainingdata_generation/README.md)
- [`Train/README.md`](./Train/README.md)

