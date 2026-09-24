# AI-Development-And-Security_FinalReport

## CT Images Classification Research Based on CNN Model

以遷移學習（Transfer Learning）方式，使用 PyTorch 與預訓練 EfficientNet-B3 模型，對胸部 CT 影像進行肺癌分類研究，涵蓋腺癌、大細胞癌、鱗狀細胞癌與正常組織四種類別，並完整實作類別不平衡處理、兩階段微調訓練、多指標模型評估與錯誤分析。

## 專案簡介

本專案使用 Kaggle 胸部 CT 掃描影像資料集，建立一個基於 CNN（EfficientNet-B3）的多類別分類模型，用於輔助肺癌類型的早期辨識：

- **adenocarcinoma**（腺癌）
- **large.cell.carcinoma**（大細胞癌）
- **squamous.cell.carcinoma**（鱗狀細胞癌）
- **normal**（正常）

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

## 模型架構與訓練策略

- **骨幹網路**：EfficientNet-B3（ImageNet 預訓練，來自 `timm`）
- **兩階段微調**：
  1. 凍結骨幹，僅訓練新接的分類頭（20 epochs）
  2. 解凍全模型，以較小學習率整體微調（25 epochs）
- **類別不平衡處理**：`WeightedRandomSampler` + 加權交叉熵損失
- **學習率排程**：`ReduceLROnPlateau`
- **早停機制**：驗證損失連續 7 個 epoch 未改善即停止訓練

## 主要成果

- 第二階段微調後，驗證集最佳準確率達 **93.06%**
- 於測試集完整輸出 Precision / Recall / F1-score（`classification_report`）與混淆矩陣
- 繪製各類別預測機率分佈、PR 曲線（含 Average Precision）與 ROC 曲線（含 AUC）
- 視覺化正確與錯誤預測樣本，協助分析模型誤判模式

## 備註

本專案僅供學術研究與學習用途。
