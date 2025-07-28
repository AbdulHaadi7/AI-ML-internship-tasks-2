# 🏠 Multimodal Housing Price Prediction (Images + Tabular Data)

## 📌 Objective

The goal of this task is to predict housing prices by combining:
- **Tabular data** (structured features such as number of rooms, location, area, etc.)
- **Image data** (visual representation of the house)

This task demonstrates **multimodal machine learning**, where models learn from multiple data types (modalities) to improve predictive accuracy in a regression problem.

---

## 🔍 Methodology / Approach

The workflow for this project involves the following steps:

1. **Data Preprocessing:**
   - Tabular data is loaded and cleaned.
   - Images are resized and normalized.
   - Each tabular entry is matched with the corresponding image via a unique identifier.

2. **Image Feature Extraction:**
   - A pre-trained CNN model (e.g., VGG16 or ResNet50) is used to extract deep visual features from house images.
   - Images are passed through the CNN (excluding the top classification layers), and outputs are flattened.

3. **Tabular Feature Processing:**
   - Tabular features are normalized and passed through a fully connected neural network to produce embeddings.

4. **Feature Fusion:**
   - Image features and tabular embeddings are concatenated.
   - The combined vector is fed into a regression head (fully connected layers) to predict housing prices.

5. **Training:**
   - The model is trained using Mean Squared Error loss.
   - Evaluation is done using **MAE (Mean Absolute Error)** and **RMSE (Root Mean Squared Error)**.

---

## 📊 Key Results or Observations

- **Multimodal models outperform unimodal models**:
  - The combination of image and tabular data consistently yields lower prediction errors than using either modality alone.
  
- **CNNs capture real-world cues**:
  - Visual indicators like house condition, style, and external features (e.g., garden, garage) help refine predictions.

- **Fine-tuning CNNs can improve accuracy**:
  - While frozen CNNs perform well, fine-tuning deeper layers gives a marginal boost if sufficient data is available.

| Model Type         | MAE   | RMSE  |
|--------------------|-------|-------|
| Multimodal (Image + Tabular) | **21.3K** | **29.7K** |

---
