# j-Wave\simulate

本資料夾提供以 [`jwave`](https://github.com/ucl-bug/jwave) 為基礎的**空氣超音波聲場模擬工具**，用以產生 PINO 模型訓練所需資料。

## 環境建議

- 建議使用 `conda` 建立獨立環境，避免與訓練環境干擾。
- 本模擬需使用 GPU。

---

## 安裝步驟

### (1) 安裝 `jwave` 所需套件

```bash
pip install -r jwave_requirement.txt
```

### (2) 安裝 `jax` 相關套件（建議 CUDA 12.4）

```bash
pip install "jax[cuda12]"==0.4.28
```

⚠️ 若無 NVIDIA GPU，請參考官方說明： [https://github.com/jax-ml/jax#installation](https://github.com/jax-ml/jax#installation)

---

## 驗證安裝是否成功

```bash
python simulate_jwave.py
```

若未報錯即表示安裝成功。

⚠️ 注意：`jax` 的環境相依性較高，目前已知成功搭配為：

- `jax==0.4.28`
- `jaxlib==0.4.28`
- `jax-cuda12-pjrt==0.4.28`
- `jax-cuda12-plugin==0.4.28`
- `jaxdf==0.2.8`

---

## 執行模擬腳本

模擬主程式為：

```bash
python simulate_jwave.py
```

---

## 詳細教學

https://hackmd.io/@benny88129/r1d4aMIAgx

