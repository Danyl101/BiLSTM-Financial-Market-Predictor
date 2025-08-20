# 📊 Financial Market Predictor — BiLSTM + Optuna

**Financial Market Predictor** is a modular system for forecasting stock market trends using a **BiLSTM** network tuned with **Optuna**. It predicts **45 days ahead** using 60-day lookback sequences and integrates a full pipeline of data preprocessing, training logs, sequence management, and evaluation.

---

## 🧠 Core Features

* 🔁 **BiLSTM Model** — Captures temporal dependencies in stock market data.
* 🧹 **Custom Preprocessing** — Sliding-window sequences, multivariate normalization with `RobustScaler`, avoiding data leakage.
* ⚙️ **Hyperparameter Optimization** — Bayesian tuning of `hidden_size`, `dropout`, `batch_size`, and `learning_rate` with Optuna.
* 💾 **Checkpointing & Logging** — Saves best models per trial with training metrics and memory usage.
* 📊 **Comprehensive Evaluation** — Computes MSE, RMSE, MAE, MAPE, and R² for robust analysis.

---

## 📁 Project Structure

```bash
FinancialMarketPredictor/
├── BiLSTM_Model/         # Model class, trainer, Optuna objective
├── BiLSTM_Preprocess/    # Dataset loader, sequence slicing, scalers
├── Checkpoints/          # Saved model weights per trial
├── Datasets/             # CSV datasets & config files
│     ├── train_scaled.csv
│     ├── val_scaled.csv
│     ├── test_scaled.csv
│     └── nifty_data.csv
└── README.md             # Documentation
```

---

## ⚙️ Setup Instructions

### 🔹 Backend (Python + PyTorch)

```bash
# 1. Clone the repo
git clone https://github.com/yourusername/BiLSTM-Financial-Market-Predictor.git

# 2. Set up virtual environment
python -m venv venv
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt
```

### 🔹 Train BiLSTM & Optimize Hyperparameters

```bash
python main.py
```

### 🔹 Evaluate & Plot Predictions

```python
from utils import plot_predictions

plot_predictions('Checkpoints/best_model_trial_0.pt')
```

---

## 📊 BiLSTM Results

| Metric   | Result   |
| -------- | -------- |
| MSE      | \~0.0299 |
| RMSE     | \~0.1730 |
| MAE      | \~0.1493 |
| MAPE     | \~6.08 % |

> Trend is smoothed and captures general market movement; peaks may have higher error due to noisy data.

### 🔍 Visualization

<img width="1504" height="853" alt="Screenshot 2025-08-06 105231" src="https://github.com/user-attachments/assets/bbdc25dd-91b6-4548-844e-7a5c9249c133" />

---

## 🧪 Workflow

1. **Prepare Dataset** → Scale & slice sequences (60-day lookback, 45-day prediction).
2. **Train Model** → BiLSTM with Optuna hyperparameter tuning.
3. **Checkpointing** → Save best models per trial.
4. **Evaluation** → Compute metrics and visualize predictions.

---

## 🛠 Tech Stack

| Layer         | Technology                  |
| ------------- | --------------------------- |
| Backend       | Python, PyTorch             |
| Hyperopt      | Optuna                      |
| Data          | Pandas, Numpy, Scikit-learn |
| Visualization | Matplotlib                  |

---


## 📄 License

MIT License — see `LICENSE` file.

---

## 🤝 Contributions

Fork the repo, open issues, or submit pull requests. Discuss major changes via GitHub issues first.

---
