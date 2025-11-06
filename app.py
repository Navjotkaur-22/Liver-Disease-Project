# app.py — Liver Disease Prediction (Streamlit, ultra-forgiving CSV + NaN-safe)
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# ---------------------------
# App Meta
# ---------------------------
st.set_page_config(page_title="Liver Disease Prediction", page_icon="🩺")
st.title("🩺 Liver Disease Prediction App")
st.write(
    "Use trained models (KNN / XGBoost) to predict the likelihood of liver disease. "
    "Supports **manual input** and **CSV upload** with auto header/delimiter fixes, schema fallback and NaN-safe preprocessing."
)

# ---------------------------
# Constants
# ---------------------------
FEATURES = [
    "Age",
    "Gender",
    "Total_Bilirubin",
    "Direct_Bilirubin",
    "Alkaline_Phosphotase",
    "Alamine_Aminotransferase",
    "Aspartate_Aminotransferase",
    "Total_Protiens",   # NOTE: dataset's spelling
    "Albumin",
    "Albumin_and_Globulin_Ratio",
]
st.caption("Required columns (order-sensitive): " + ", ".join(FEATURES))

# ---------------------------
# Model loading
# ---------------------------
@st.cache_resource
def load_models():
    models = {}
    if os.path.exists("best_knn_liver_model.joblib"):
        models["KNN"] = joblib.load("best_knn_liver_model.joblib")
    if os.path.exists("best_xgb_liver_model.joblib"):
        models["XGBoost"] = joblib.load("best_xgb_liver_model.joblib")
    if os.path.exists("scaler_for_knn.joblib"):
        models["SCALER"] = joblib.load("scaler_for_knn.joblib")
    return models

MODELS = load_models()
AVAILABLE = [m for m in ["KNN", "XGBoost"] if m in MODELS]
if not AVAILABLE:
    st.error(
        "Model files not found. Place **best_knn_liver_model.joblib** or **best_xgb_liver_model.joblib** "
        "next to this app.py (and **scaler_for_knn.joblib** for KNN)."
    )
    st.stop()

model_choice = st.radio("Select model", AVAILABLE, horizontal=True)

# ---------------------------
# Header normalization (fix variants/typos)
# ---------------------------
def normalize_headers(df: pd.DataFrame) -> pd.DataFrame:
    def norm(s: str) -> str:
        return (
            str(s).strip()
            .lower()
            .replace(" ", "_")
            .replace("-", "_")
            .replace("/", "_")
        )

    mapping = {
        "age": "Age",
        "gender": "Gender",
        "total_bilirubin": "Total_Bilirubin",
        "direct_bilirubin": "Direct_Bilirubin",
        "alkaline_phosphotase": "Alkaline_Phosphotase",   # common misspelling
        "alkaline_phosphatase": "Alkaline_Phosphotase",   # variant
        "alamine_aminotransferase": "Alamine_Aminotransferase",
        "aspartate_aminotransferase": "Aspartate_Aminotransferase",
        "total_proteins": "Total_Protiens",               # map to dataset spelling
        "total_protiens": "Total_Protiens",
        "albumin": "Albumin",
        "albumin_and_globulin_ratio": "Albumin_and_Globulin_Ratio",
        "agr": "Albumin_and_Globulin_Ratio",
        "a_g_ratio": "Albumin_and_Globulin_Ratio",
    }

    return df.rename(columns={c: mapping.get(norm(c), c) for c in df.columns})

# ---------------------------
# Super-forgiving preprocess
# ---------------------------
def preprocess(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Super-forgiving preprocessing:
    - Normalize headers & strip spaces
    - Drop fully-empty columns and index-like columns
    - If cols >= 10, auto-pick first 10 after cleanup and force-assign FEATURES
    - Map Gender {Male->1, Female->0}
    - Coerce numerics & impute medians
    - Scale numerics for KNN if scaler present
    """
    df = df_raw.copy()

    # 1) basic cleanup
    df.columns = [str(c).strip() for c in df.columns]
    df = normalize_headers(df)

    # Drop fully-empty columns (all NaN or all empty strings)
    drop_these = []
    for c in df.columns:
        col = df[c]
        if col.isna().all():
            drop_these.append(c)
        else:
            if (col.astype(str).str.strip() == "").all():
                drop_these.append(c)
    if drop_these:
        df = df.drop(columns=drop_these, errors="ignore")

    # Drop obvious index-like first column (Unnamed / index / blank)
    if df.shape[1] >= 1:
        first = str(df.columns[0]).lower()
        if first.startswith("unnamed") or first in ("index", ""):
            df = df.drop(columns=[df.columns[0]], errors="ignore")

    # 2) ensure schema
    missing = [c for c in FEATURES if c not in df.columns]
    if missing:
        # if we still have >= 10 columns, force-pick the first 10 after cleanup
        if df.shape[1] >= len(FEATURES):
            df = df.iloc[:, :len(FEATURES)].copy()
            df.columns = FEATURES
        else:
            raise ValueError(f"Missing required columns: {missing}")

    # 3) Gender mapping
    if "Gender" in df.columns and df["Gender"].dtype == object:
        df["Gender"] = (
            df["Gender"].astype(str).str.strip().map({
                "Male": 1, "Male ": 1, "M": 1, "male": 1,
                "Female": 0, "female": 0, "F": 0
            })
        )

    # 4) Select in order
    X = df[FEATURES].copy()

    # 5) Coerce numerics & impute medians
    for col in FEATURES:
        if col != "Gender":
            X[col] = pd.to_numeric(X[col], errors="coerce")
    num_cols = X.select_dtypes(include=[np.number]).columns
    for c in num_cols:
        X[c] = X[c].fillna(X[c].median())
    X = X.fillna(X.median(numeric_only=True))

    # 6) Scale for KNN if scaler present
    if model_choice == "KNN" and "SCALER" in MODELS:
        X[num_cols] = MODELS["SCALER"].transform(X[num_cols])

    return X

# ---------------------------
# Smart CSV reader (forgiving)
# ---------------------------
def smart_read_csv(uploaded_file):
    """
    Tries:
    1) Normal read
    2) Auto-detect delimiter
    3) Headerless read -> set FEATURES; drop accidental index if 11 cols; drop trailing empty cols
    """
    def normal(f):
        return pd.read_csv(f)

    def autosep(f):
        return pd.read_csv(f, sep=None, engine="python")

    def headerless(f):
        df = pd.read_csv(f, header=None)
        # drop trailing empty cols (common in Excel)
        empty_cols = []
        for c in df.columns:
            col = df[c]
            if col.isna().all() or (col.astype(str).str.strip() == "").all():
                empty_cols.append(c)
        if empty_cols:
            df = df.drop(columns=empty_cols, errors="ignore")

        # drop accidental leading index
        if df.shape[1] == len(FEATURES) + 1:
            df = df.drop(columns=df.columns[0])

        if df.shape[1] < len(FEATURES):
            raise ValueError(f"Headerless CSV must have at least {len(FEATURES)} cols, found {df.shape[1]}")
        if df.shape[1] > len(FEATURES):
            df = df.iloc[:, :len(FEATURES)]
        df.columns = FEATURES
        return df

    for reader in (normal, autosep, headerless):
        uploaded_file.seek(0)
        try:
            df = reader(uploaded_file)
            return df
        except Exception:
            continue

    raise ValueError("Unable to parse CSV")

# ---------------------------
# UI
# ---------------------------
tab1, tab2 = st.tabs(["🔢 Manual Input", "📄 CSV Upload"])

with tab1:
    st.subheader("Enter patient features")

    col1, col2 = st.columns(2)
    Age = col1.number_input("Age", min_value=0, max_value=120, value=45, step=1)
    Gender = col2.selectbox("Gender", ["Male", "Female"])

    Total_Bilirubin = col1.number_input("Total_Bilirubin", min_value=0.0, value=1.0, step=0.1, format="%.2f")
    Direct_Bilirubin = col2.number_input("Direct_Bilirubin", min_value=0.0, value=0.3, step=0.1, format="%.2f")

    Alkaline_Phosphotase = col1.number_input("Alkaline_Phosphotase", min_value=0, value=200, step=1)
    Alamine_Aminotransferase = col2.number_input("Alamine_Aminotransferase", min_value=0, value=30, step=1)
    Aspartate_Aminotransferase = col1.number_input("Aspartate_Aminotransferase", min_value=0, value=35, step=1)

    Total_Protiens = col2.number_input("Total_Protiens", min_value=0.0, value=6.5, step=0.1, format="%.2f")
    Albumin = col1.number_input("Albumin", min_value=0.0, value=3.5, step=0.1, format="%.2f")
    Albumin_and_Globulin_Ratio = col2.number_input("Albumin_and_Globulin_Ratio", min_value=0.0, value=1.0, step=0.1, format="%.2f")

    if st.button("Predict", use_container_width=True):
        row = {
            "Age": Age,
            "Gender": 1 if Gender == "Male" else 0,
            "Total_Bilirubin": Total_Bilirubin,
            "Direct_Bilirubin": Direct_Bilirubin,
            "Alkaline_Phosphotase": Alkaline_Phosphotase,
            "Alamine_Aminotransferase": Alamine_Aminotransferase,
            "Aspartate_Aminotransferase": Aspartate_Aminotransferase,
            "Total_Protiens": Total_Protiens,
            "Albumin": Albumin,
            "Albumin_and_Globulin_Ratio": Albumin_and_Globulin_Ratio,
        }
        try:
            X = preprocess(pd.DataFrame([row]))
            model = MODELS[model_choice]
            pred = int(model.predict(X)[0])
            label = "Liver Disease" if pred == 1 else "No Disease"
            st.success(f"Prediction: **{label}**")
            if hasattr(model, "predict_proba"):
                proba = float(model.predict_proba(X)[0, 1])
                st.write(f"Probability (class 1): **{proba:.2f}**")
        except Exception as e:
            st.error(f"Error during prediction: {e}")

with tab2:
    st.subheader("Upload CSV (auto-fix headers/index)")
    debug = st.checkbox("Show debug info", value=True)
    file = st.file_uploader("Select CSV file", type=["csv"])

    if file is not None:
        try:
            raw = smart_read_csv(file)
            if debug:
                st.write("🔹 Raw detected columns:", list(raw.columns))
                st.write("🔹 Raw head:", raw.head())

            X = preprocess(raw)

            if debug:
                st.write("🔹 Final FEATURES used:", FEATURES)
                st.write("🔹 Final dtypes:", {k: str(v) for k, v in X.dtypes.items()})
                st.write("🔹 NaN counts:", X.isna().sum().to_dict())
                st.write("🔹 Final sample to model:", X.head())

            model = MODELS[model_choice]
            preds = model.predict(X)
            out = raw.copy()
            out["prediction"] = preds
            if hasattr(model, "predict_proba"):
                out["proba_1"] = model.predict_proba(X)[:, 1]

            st.write("Sample predictions:", out.head())
            st.download_button(
                label="Download predictions CSV",
                data=out.to_csv(index=False).encode("utf-8"),
                file_name="liver_predictions.csv",
                mime="text/csv"
            )
        except Exception as e:
            st.error(f"Failed to score CSV: {e}")

st.info(
    "Notes: Gender mapped as Male=1, Female=0. Numeric NaNs filled with medians. "
    "KNN applies saved scaler if present."
)

     
