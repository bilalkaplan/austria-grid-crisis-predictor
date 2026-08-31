# AI-Based Grid Flexibility and Crisis Prediction System ⚡

![Grid Operations](https://img.shields.io/badge/Domain-Energy%20Grid-blue)
![PyTorch](https://img.shields.io/badge/Framework-PyTorch-ee4c2c)
![MIT LIDS](https://img.shields.io/badge/Inspired%20by-MIT%20LIDS-gray)
![Status](https://img.shields.io/badge/Status-Completed-success)

A pro-active, deep learning-based decision support system designed to predict critical load thresholds and grid flexibility crises **24 hours in advance**. The project utilizes **Spatial Awareness** and custom **Smoothness Penalty** optimization to prevent model collapse in highly chaotic and anomalous grid environments.

This project was developed as part of a Microsoft Internship program.

---

## 📖 The Problem: Renewable Energy and Grid Instability
The integration of renewable energy sources inherently brings meteorological uncertainties. Sudden drops in solar generation or wind cuts can drastically disrupt the supply-demand balance, directly threatening the physical flexibility of modern power grids. 

Focusing on the **Austrian National Grid**, this system acts as a proactive early warning mechanism for operational teams, predicting future energy consumption and identifying critical load thresholds that strain grid capacity.

## 🧠 Academic Vision & Spatial Awareness
This project goes beyond standard sequence memorization. It integrates principles of **Spatial Awareness**, heavily inspired by the MIT LIDS paper published at NeurIPS (Oct 2025):

> **"Smooth Sailing: Lipschitz-Driven Uncertainty Quantification for Spatial Association"**  
> *David R. Burt, Renato Berlinghieri, Stephen Bates, Tamara Broderick (MIT LIDS)*

Standard models often suffer from "spatial blindness"—working perfectly in one region (e.g., sunny weather) but collapsing in another (e.g., snowy conditions). To prove that our architecture learns universal grid rules rather than memorizing the Austrian grid, we subjected it to a chaotic **"Region B" Stress Test** (30% less wind, 15% less load, and added synthetic noise).

## ⚙️ Architecture and Technical Highlights

### 1. Data Ecosystem (ENTSO-E)
* Extracted hourly solar, wind, and load metrics from the European **ENTSO-E** network.
* **Feature Engineering:** Integrated Sine/Cosine transformations to capture daily and monthly cyclical temporal features.
* **Anomaly Repair:** Applied temporal interpolation to rebuild mathematically invalid sensor readings (e.g., regional consumption erroneously dropping below 4000 MW).

### 2. Multi-Sequence PyTorch Models
* Benchmarked **LSTM**, **Bi-LSTM**, and **GRU** architectures simultaneously to capture short and long-term dependencies.
* **Sliding Window:** Analyzes 48 hours of historical data (3D tensor) to predict the exact profile of the next 24 hours.
* **Cross-Validation:** 3-Fold `TimeSeriesSplit` to prevent Data Leakage.
* **Optimization:** Utilized `ReduceLROnPlateau` and Early Stopping (85% Train, 15% Validation split).

### 3. Physical Smoothness Penalty (Custom Loss)
Inspired by Lipschitz continuity, a custom mathematical **Smoothness Penalty** layer was developed. Standard MSE loss functions often produce physically impossible sudden jumps in chaotic environments. This custom penalty mathematically punishes erratic, unphysical hour-to-hour transitions, forcing the model to respect real-world grid inertia.

## 📊 Performance and Operational Impact

On the highly chaotic Region B test set, the models achieved the following **WAPE (Weighted Absolute Percentage Error)** scores, staying loyal to physical grid reality:
* 🏆 **GRU:** 12.36%
* 🥈 **Bi-LSTM:** 12.43%
* 🥉 **LSTM:** 16.01%

### 🚨 Crisis Prediction Success
A **"Grid Flexibility Crisis"** is operationally defined as the moment the net grid load exceeds **6000 MW**. The optimized GRU model predicts these critical moments 24 hours in advance with:
* **Recall (Crisis Detection):** 77.44%
* **F1-Score:** 78.59%

## 📂 Project Structure

All Jupyter notebooks have been converted to English and logically separated into the following pipeline:

1. `1-data_loading_inspection.ipynb`: Data extraction and initial ENTSO-E inspection.
2. `2-creating_test_sets.ipynb`: Validation and test set generation.
3. `3-model_creation.ipynb`: PyTorch architecture definitions.
4. `4-initial_predictions.ipynb`: Baseline model evaluations.
5. `5-LSTM_GRU_Error_Score_Comparison.ipynb`: Metric comparisons.
6. `6-Watt_Comparison.ipynb`: Absolute wattage deviation analysis.
7. `7-Comprehensive_Model_Comparison.ipynb`: Complete metric compilation (MSE, RMSE, MAE, MAPE).
8. `8-Austria_Final_Visualization.ipynb`: Baseline visualizations.
9. `9-Austria_Solar_Wind_Load_MAE_WAPE.ipynb`: Renewable capacity impact.
10. `10-dataset_analysis.ipynb`: Deep statistical data review.
11. `11-Daily_Net_Load_and_Grid_Flexibility_Forecast.ipynb`: Core 24-hour forecasting logic.
12. `12-GRU_CROSS_VALIDATION_ANALYSIS.ipynb`: GRU TimeSeriesSplit validation.
13. `13-time_encoding_analysis.ipynb`: Sine/Cosine feature impact.
14. `14-LSTM_Cross_Validation.ipynb`: LSTM TimeSeriesSplit validation.
15. `15-BiLSTM_CROSS_VALIDATION.ipynb`: Bi-LSTM TimeSeriesSplit validation.
16. `16-net_load_flexibility_forecast.ipynb`: Final load and flexibility integration.
17. `17-Final_Spatial_Awareness.ipynb`: **The Region B Stress Test & Smoothness Penalty.**
18. `18-forecast.ipynb`: Final consolidated forecasting pipeline.

## 🚀 Future Work
* **Meteorological Integration:** Feeding real-time satellite radar and weather APIs directly into the prediction stream.
* **National Scaling:** Adapting the spatial awareness infrastructure to other European zones and the Turkish National Grid (TEİAŞ).
* **Autonomous Action:** Evolving the system from a "crisis warning" engine into an autonomous grid brain capable of proposing Automatic Load Shedding commands.

---
*Developed by Bilal Kocakaplan*
