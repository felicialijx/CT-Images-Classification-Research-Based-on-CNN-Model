# AI-Development-And-Security_FinalReport

## CT Images Classification Research Based on CNN Model

以遷移學習（Transfer Learning）方式，使用 PyTorch 與預訓練 EfficientNet-B3 模型，對 Kaggle 胸部 CT 影像進行肺癌分類。涵蓋腺癌、大細胞癌、鱗狀細胞癌與正常組織四種類別，並完整實作類別不平衡處理、兩階段微調訓練、多指標模型評估與錯誤分析。

## 資料來源

- **資料集**：[Chest CT-Scan Images](https://www.kaggle.com/datasets/mohamedhanyyy/chest-ctscan-images)（Kaggle）
- 資料集包含 train / valid / test 三個分割，原始類別資料夾名稱不一致，專案中已統一整理為標準類別名稱

## 分析流程

| 步驟 | 內容 |
|------|------|
| 1 | 環境與訓練參數設定（影像尺寸、batch size、學習率、epoch 數等） |
| 2 | 整理資料夾結構，統一各類別命名 |
| 3 | 資料載入與影像前處理（資料增強、標準化） |
| 4 | 檢查訓練集類別分佈，以 `WeightedRandomSampler` 處理類別不平衡 |
| 5 | 載入預訓練 EfficientNet-B3 模型，替換分類頭 |
| 6 | 定義加權交叉熵損失函數與訓練函數（含早停機制） |
| 7 | 第一階段訓練：凍結骨幹層，僅訓練分類頭 |
| 8 | 第二階段訓練：解凍全模型進行微調（Fine-tuning） |
| 9 | 測試集評估：分類報告、混淆矩陣、預測機率分佈、PR / ROC 曲線、預測樣本與錯誤分析視覺化 |

```text
  Kaggle：Chest CT-Scan Images
            │
            ▼
  ① 資料整理
     統一 4 類別名稱 → Train 613 / Valid 72 / Test 315
            │
            ▼
  ② 資料預處理
     ├─ Train       RandomResizedCrop、翻轉、旋轉 ±15°、ColorJitter
     └─ Valid/Test  Resize、CenterCrop、Normalize
            │
            ▼
  ③ 類別不平衡處理
     WeightedRandomSampler ＋ 加權 CrossEntropyLoss
            │
            ▼
  ④ 建立模型
     EfficientNet-B3（ImageNet 預訓練）→ 分類頭改為 4 類
            │
            ▼
  ⑤ 兩階段訓練
     ├─ Stage 1  lr=2e-3, 20 epochs  凍結骨幹，只訓練分類頭
     └─ Stage 2  lr=2e-4, 25 epochs  解凍全部層微調
            │
            │  每個 epoch：train → valid → ReduceLROnPlateau
            │  ├─ val loss 改善     → 儲存最佳權重
            │  └─ 連續 7 次未改善   → 早停
            ▼
  ⑥ 測試集評估（315 張）
     ├─ Classification Report（Accuracy 93.33%）
     ├─ Confusion Matrix
     ├─ 預測機率分佈
     ├─ PR / ROC 曲線
     └─ 錯誤樣本分析
```

## 主要成果

- 第二階段微調後，驗證集最佳準確率達 **93.06%**
- 於測試集完整輸出 Precision / Recall / F1-score（`classification_report`）與混淆矩陣
- 繪製各類別預測機率分佈、PR 曲線（含 Average Precision）與 ROC 曲線（含 AUC）
- 視覺化正確與錯誤預測樣本，協助分析模型誤判模式

| 類別 | Precision | Recall | F1-score |
|------|-----------|--------|----------|
| adenocarcinoma | 0.9106 | 0.9333 | 0.9218 |
| large.cell.carcinoma | 0.8793 | 1.0000 | 0.9358 |
| normal | 1.0000 | 0.9815 | 0.9907 |
| squamous.cell.carcinoma | 0.9630 | 0.8667 | 0.9123 |

## 備註

本專案僅供學術研究與學習用途。
