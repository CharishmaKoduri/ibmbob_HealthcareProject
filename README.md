# 🏥 Healthcare Prediction System

A full-stack machine learning application that predicts **patient test results**, **hospital billing costs**, and **admission type** from demographic and clinical data — built entirely in Python.

---

## 📂 Dataset

- **File:** `healthcare_dataset.csv`
- **Records:** 55,500 patients
- **Source:** [Kaggle — Healthcare Dataset](https://www.kaggle.com/datasets/prasad22/healthcare-dataset)
- **Columns:** Name, Age, Gender, Blood Type, Medical Condition, Date of Admission, Doctor, Hospital, Insurance Provider, Billing Amount, Room Number, Admission Type, Discharge Date, Medication, Test Results

---

## 🎯 Prediction Models

| # | Model | Target | Algorithm |
|---|---|---|---|
| 1 | Test Result Classifier | Normal / Abnormal / Inconclusive | Random Forest |
| 2 | Billing Estimator | Hospital Billing Amount ($) | Gradient Boosting Regressor |
| 3 | Admission Risk Classifier | Elective / Emergency / Urgent | Random Forest |

---

## 🗂️ Project Structure

```
healthcare-prediction/
├── healthcare_dataset.csv        # Source data
├── app.py                        # Streamlit frontend
├── train_model.py                # Model training script
├── HealthcarePrediction.ipynb   # Jupyter Notebook (full pipeline)
├── requirements.txt              # Python dependencies
├── README.md                     # This file
├── KoduriCharishma_ProjectReport.docx   # Full project documentation
└── models/
    ├── test_result_model.pkl
    ├── billing_model.pkl
    ├── admission_model.pkl
    ├── encoders.pkl
    └── meta.pkl
```

---

## 🛠️ Technologies Used

| Category | Library / Tool |
|---|---|
| **Frontend** | Streamlit |
| **Visualisation** | Plotly, Matplotlib, Seaborn |
| **ML Models** | scikit-learn (Random Forest, Gradient Boosting) |
| **Data Wrangling** | Pandas, NumPy |
| **Notebook** | Jupyter / ipykernel |
| **Language** | Python 3.10+ |

---

## ⚙️ Setup & Run Instructions

### 1. Clone / download the project
```bash
git clone <your-repo-url>
cd healthcare-prediction
```

### 2. Create a virtual environment (recommended)
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Train the ML models (first time only)
```bash
python train_model.py
```
This generates all `.pkl` files inside the `models/` folder.

### 5. Launch the Streamlit app
```bash
streamlit run app.py
```
Open your browser at **http://localhost:8501**

### 6. Run the Jupyter Notebook (optional)
```bash
jupyter notebook HealthcarePrediction.ipynb
```

---

## 📱 App Pages

| Page | Description |
|---|---|
| 🏠 Home | KPI cards, dataset snapshot, condition distribution |
| 📊 EDA Dashboard | 10+ interactive charts exploring the dataset |
| 🔬 Test Result Predictor | Live prediction with confidence probabilities |
| 💰 Billing Estimator | Cost prediction with benchmark comparison |
| 🚑 Admission Risk | Predict admission type from patient details |
| 📈 Model Performance | Accuracy metrics, confusion matrices, feature importance |

---

## 📊 Model Performance

| Model | Metric | Value |
|---|---|---|
| Test Result Classifier | Accuracy | ~43% |
| Admission Type Classifier | Accuracy | ~44% |
| Billing Regressor | MAE | ~$12,197 |

> **Note:** The dataset uses synthetically randomised labels, resulting in near-random-chance accuracy (~43% vs 33% random baseline for 3 classes). The pipeline is production-ready and will improve significantly with real clinical data.

---

## 👤 Author

**Koduri Charishma**  

---

## 📄 License

This project is submitted for academic purposes.
