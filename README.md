# Diabetes Prediction Model

This project leverages machine learning to predict whether a patient is likely to develop diabetes based on diagnostic measurements. The model is trained on the Pima Indians Diabetes dataset and includes comprehensive evaluation metrics and visualizations.

## Live Demo

Try the interactive app here:  
 [Diabetes Prediction Web App](https://diabetes-prediction-nsut-satwik.streamlit.app/)


## Dataset

The dataset used is the [Pima Indians Diabetes Dataset](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database). It includes 768 samples and 8 clinical features commonly associated with diabetes.

**Target Variable:**
- `Outcome`: 
  - `1` → Diabetic  
  - `0` → Non-diabetic

**Features:**
- `Pregnancies`
- `Glucose`
- `BloodPressure`
- `SkinThickness`
- `Insulin`
- `BMI`
- `DiabetesPedigreeFunction`
- `Age`

---

##  Model Details

- **Algorithm**: (insert model type, e.g. `Logistic Regression`)
- **Data Scaler**: StandardScaler
- **Train/Test Split**: 80/20
- **Evaluation Metrics**:
  - Accuracy (Train/Test)
  - Confusion Matrix
  - ROC Curve & AUC
  - Classification Report (Precision, Recall, F1-score,source)

**Model Files:**
- `models/diabetes_model.pkl` – Trained machine learning model
- `models/scaler.pkl` – Scaler used to normalize input features


## Author

**Satwik**  
- 🔗 GitHub: [Satwik200500](https://github.com/Satwik200500)  
- 📧 Email: satwik.ug23@nsut.ac.in  
- 🌐 LinkedIn: [satwik-b65a30297](https://www.linkedin.com/in/satwik-b65a30297)




