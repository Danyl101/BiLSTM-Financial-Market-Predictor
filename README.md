### # Financial Market Predictor (BiLSTM + Optuna)

**Predict 45 days ahead in normalized stock data using a Bi‑LSTM optimized with Optuna. Built with full modular architecture including dataset loaders, training logs, sequence prep, and evaluation — part of a larger system with scraper and future BERT integration.**

---

### 🚀 Overview

This project is a core module of a larger pipeline that includes:

* A web scraper + React frontend for collecting article keywords and target websites
* A **Bi‑LSTM time‑series forecasting engine** tuned via *Optuna*
* Commit, snapshot, and hyperparameter logs to ensure reproducibility

The goal: **predict future stock movement 45 days ahead** using past 60-day lookback sequences.

---

### 📁 Project Structure

```
├── /Datasets/                 # Scaled CSVs & goodlist config files
│     ├── train_scaled.csv
│     ├── val_scaled.csv
│     ├── test_scaled.csv
│     └── config.json
├── /lstm_dataload.py          # PyTorch Dataset & sequence logic
├── /lstm_model.py             # Model class and objective() for Optuna
├── /main.py                   # Train + optimize hyperparams, optional manual runs
├── /utils.py                  # Helper functions: logging, metrics, plotting
├── /Checkpoints/              # Trained model checkpoints saved by trial
└── README.md                  # (This file)
```

---

### 🧠 Key Features

* **Sliding-window, multivariate forecasting**: Input Open‑High‑Low‑Volume features for 60 day lookbacks, predict 45 days ahead for next close.
* **Robust scaling pipeline**: Uses `RobustScaler` fit only on training subsets to avoid data leakage.
* **Optuna hyperparameter tuning**: Searches across `hidden_size`, `dropout`, `batch_size`, and `learning_rate` for best validation MSE.
* **Custom loss and checkpoint handling**: Model training logs memory usage and stores best checkpoints per-trial.
* **Thorough evaluation**: Tracks metrics—MSE, RMSE, MAE, MAPE, R²—for comprehensive audit and validation.

---

### 📊 Metrics & Results

| Metric   | Result   |
| -------- | -------- |
| MSE      | \~0.0550 |
| RMSE     | \~0.2345 |
| MAE      | \~0.2030 |
| MAPE     | \~8.31 % |
| R² score | \~0.0449 |

* **Vision**: realistic long-horizon forecasting (45 days)
* **Validation**: model does not overfit despite chaotic financial signals
* **Plot Behavior**: smoothed trend following, with error in peaks expected from noisy data

---

### 🛠️ How To Run (Colab / Local)

1. **Prepare data**:

   ```bash
   python lstm_dataextract.py
   python lstm_dataprocess.py
   ```
2. **Train/Bayesian tune**:

   ```bash
   python main.py
   ```
3. **Evaluate and plot**:

   ```py
   from utils import plot_predictions
   plot_predictions('Checkpoints/best_model_trial_0.pt')
   ```
4. (Optional) Inference time: run on new data for real predictions.

---

### 🔍 Visualization

<img width="1504" height="853" alt="BiLSTM Graph" src="https://github.com/user-attachments/assets/807d978c-b1cf-4a4e-8022-a3fab7ed5746" />

---

### 🧪 Best Practices

* **Use time-based, non-shuffled splits** for train/validation/test.
* **Fit the scaler only on training data**, then transform eval/test.
* Always log CPU/GPU RAM usage to catch memory issues mid-training.
* Make `prediction_gap` configurable—then compare gap = 20, 45, 60 to test generalization.
* Save model checkpoints per-trial for reproducibility and future retraining.

---

### 🛰️ Roadmap & Integration

This repo feeds into a greater system currently under development:

* **Web scraper & UI**: Automates news collection and keyword management.
* **BERT-based NLP pipeline**: Classify or fact‑check scraped news, integrate sentiment into time-series modeling.
* **Hybrid forecasting stack**: Combine BiLSTM (trend) + BERT (news) into a meta‑model for stronger joint insight.

---

### 📜 Academic & Publication Worthy?

Absolutely. The system has:

* A clear experimental log with hyperparameter milestones.
* Progressive comparison: TCN → LSTM → BiLSTM.
* Handling of real-world issues: data drift, leakage, distribution mismatch.
* Metrics and qualitative analysis across forecasting horizons, ideal for a technical report or paper.

---

### 📄 License

MIT or chosen open license.

---

You can copy this template into your GitHub repo and adjust details like file names, add links to demo plots, and mention the repository hierarchy (e.g. monorepo or split). Let me know if you’d like help refining the visual elements or writing matching READMEs for your other scrap‑based modules.
