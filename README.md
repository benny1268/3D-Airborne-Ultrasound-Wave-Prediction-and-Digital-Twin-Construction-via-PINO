# 🌀 3D Airborne Ultrasound Wave Prediction and Digital Twin Construction via PhysicsInformed Neural Operators (PINO) 

本專案提供一套完整流程，用於訓練 **空氣超音波陣列物理模擬的 PINO 模型**，包含：

- 聲場模擬（使用 [`jwave`](https://github.com/ucl-bug/jwave)）
- AI 模型訓練（使用 [NVIDIA PhysicsNeMo](https://developer.nvidia.com/physicsnemo)）
- 建議兩個資料夾專案，分開建立conda環境來執行，避免環境干擾
---

## 1. `Trainingdata_generation/`：聲場模擬資料生成

1. 此資料夾包含以 [`jwave`](https://github.com/ucl-bug/jwave) 進行聲場模擬的腳本。  
2. `jwave` 是一個基於 [`JAX`](https://github.com/jax-ml/jax) 的 CUDA 加速超音波模擬工具。

---

### 安裝步驟

#### (1) 安裝 jwave 相關套件：
```bash
pip install -r jwave_requirement.txt
```

#### (2) 安裝 jax 套件：
```bash
pip install "jax[cuda12]"==0.4.28 
```
⚠️ 如果沒有 NVIDIA GPU，請至 JAX 官方頁面選擇適合的安裝方式：
👉 https://github.com/jax-ml/jax#installation

---

### 驗證：
```bash
python simulate_jwave.py
```
若未報錯即表示安裝成功。
⚠️ JAX版本很容易有問題，目前還找不到比較好的方式解決，測試過後發現 cuda12.4版本下jax, jaxlib, jax-cuda12-pjrt, jax-cuda12-plugin==0.4.28 跟 jaxdf=0.2.8 成功機率高高

---


### 執行模擬程式：

主要模擬腳本為：

```bash
python simulate_jwave.py
```
可在程式中的 main 區塊調整模擬參數（如模擬時間等）。
執行後會在指令的資料夾生成訓練資料

---


## 2. `Train/`：PINO 模型訓練

此資料夾包含基於 NVIDIA PhysicsNeMo 的 AI 模型訓練腳本，使用由 jwave 生成的資料進行超音波場的時序預測模型訓練。

---


### 安裝步驟

#### (1) 安裝核心套件：
```bash
pip install -r nv_physicsnemo_requirement.txt
```
#### (2) 安裝 PhysicsNeMo-Sym 模組（重要）：
```bash
pip install nvidia-physicsnemo-sym --no-build-isolation
```
若遇到安裝問題，請參考官方教學文件： https://docs.nvidia.com/deeplearning/physicsnemo/getting-started/index.html

---

### 驗證：
```bash
python physicsnemo_check.py
```
若輸出torch.Size([128, 64])即表示安裝成功。

---

### 執行訓練程式：
主要訓練腳本為：
```bash
python train.py
```
主要可以調整的參數都可以於conf/config_pino.yaml中設定
