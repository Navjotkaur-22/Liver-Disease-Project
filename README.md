# 🩺 Liver Disease Prediction App  

An interactive **Machine Learning Web App** built with **Streamlit**, that predicts whether a patient is likely to have liver disease based on clinical parameters.  
This project demonstrates a complete **end-to-end Data Science workflow** — from dataset preprocessing to model tuning, validation, and live deployment.  

---

## 🌐 Live Demo  

<p align="center">
<a href="https://liver-disease-project-9gfszs3dclhwspmr6zedsb.streamlit.app/" target="_blank">
<img src="https://img.shields.io/badge/🚀_Launch_App-Streamlit-orange?style=for-the-badge"/>
</a>
</p>

---

## 🧠 Project Overview  

This application enables users to:  
- 🧾 **Enter medical parameters manually** or  
- 📂 **Upload CSV files** for batch predictions  
- 🔧 Automatically handle missing headers, NaN values & scaling  
- ⚙️ Predict outcomes using a tuned **XGBoost Classifier**  
- 🌈 View clean, interpretable prediction results with probability confidence  

---

## 🧩 Key Features  

✅ Manual input & batch CSV prediction  
✅ NaN-safe preprocessing & encoding  
✅ **SMOTE** balancing for improved recall  
✅ Hyperparameter tuning via **GridSearchCV**  
✅ Stratified K-Fold validation for robust performance  
✅ Live deployment via **Streamlit Cloud**  

---

## 🧮 Input Parameters  

| Feature | Description |
|----------|-------------|
| Age | Patient age (in years) |
| Gender | Male / Female |
| Total_Bilirubin | Total bilirubin level |
| Direct_Bilirubin | Direct bilirubin level |
| Alkaline_Phosphotase | Enzyme level |
| Alamine_Aminotransferase | Enzyme level |
| Aspartate_Aminotransferase | Enzyme level |
| Total_Protiens | Total proteins in blood |
| Albumin | Albumin level |
| Albumin_and_Globulin_Ratio | Albumin to globulin ratio |

---

## ⚙️ Model Workflow  

1️⃣ **Data Preprocessing** → Missing value handling & label encoding  
2️⃣ **Imbalance Correction** → Applied **SMOTE**  
3️⃣ **Feature Scaling & Model Training** → XGBoost Classifier  
4️⃣ **Hyperparameter Tuning** → GridSearchCV with Stratified 5-Fold CV  
5️⃣ **Performance Evaluation** → Accuracy, F1-score, ROC-AUC, and Confusion Matrix  
6️⃣ **Model Export** → Saved final pipeline using Joblib  
7️⃣ **Deployment** → Streamlit app for real-time inference  

---

## 📊 Model Performance  

| Metric | Score |
|---------|-------|
| Accuracy | ~91% |
| F1-Score | High |
| ROC-AUC | Excellent |
| Validation | 5-Fold Stratified CV |

*(Exact metrics may vary slightly across runs depending on resampling.)*

---

## 🧰 Tech Stack  

- **Python 3.11+**  
- **Streamlit** (Web Framework)  
- **Pandas**, **NumPy** (Data Handling)  
- **Scikit-learn**, **XGBoost**, **SMOTE** (ML Workflow)  
- **Matplotlib**, **Seaborn** (Visualizations)  
- **Joblib** (Model Serialization)  
- **GitHub + Streamlit Cloud** (Deployment)

---

## 🚀 How to Run Locally  

```bash
# Clone this repository
git clone https://github.com/Navjotkaur-22/Liver-Disease-Project.git

cd Liver-Disease-Project

# Install dependencies
pip install -r requirements.txt

# Run the Streamlit app
streamlit run app.py

✨ Author

👩🏻‍💻 Navjot Kaur
MSc (IT) | Certified Data Scientist | Streamlit Developer
📍 Jalandhar, Punjab
📧 Email: nkaur4047@gmail.com

<p align="center"> <a href="https://github.com/Navjotkaur-22" target="_blank"><img src="https://img.shields.io/badge/GitHub-Navjotkaur--22-black?logo=github&style=for-the-badge"/></a> <a href="https://www.linkedin.com/in/navjot-kaur-b61aab299/" target="_blank"><img src="https://img.shields.io/badge/LinkedIn-Navjot_Kaur-blue?logo=linkedin&style=for-the-badge"/></a> <a href="https://www.upwork.com/freelancers/~01b30aa09d478b524c" target="_blank"><img src="https://img.shields.io/badge/Upwork-Hire_Me-brightgreen?logo=upwork&style=for-the-badge"/></a> </p> ```
