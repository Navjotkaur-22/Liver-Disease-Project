# app_streamlit.py — CSV Upload First, Professional Bulk Prediction UI

import io
import numpy as np
import pandas as pd
import streamlit as st
import joblib

st.set_page_config(page_title="Skin Disorder Classification", layout="wide")

# ---------- Load Artifacts ----------
@st.cache_resource
def load_artifacts():
    model = joblib.load("artifact/model.pkl")
    scaler = joblib.load("artifact/scaler.pkl")
    return {"model": model, "scaler": scaler}

BUNDLE = load_artifacts()

def get_feature_names():
    # Prefer names saved inside scaler (sklearn ≥1.0 when fit on DataFrame)
    scaler = BUNDLE["scaler"]
    if hasattr(scaler, "feature_names_in_"):
        return list(scaler.feature_names_in_)
    # Fallback: 34 unnamed features
    return [f"feature_{i+1}" for i in range(34)]

FEATURE_COLUMNS = get_feature_names()
N_FEATURES = len(FEATURE_COLUMNS)

# ---------- Predict Helpers ----------
def transform_and_predict(df_features: pd.DataFrame):
    # Ensure only required columns and correct order
    X = df_features[FEATURE_COLUMNS].astype(float)
    Xs = BUNDLE["scaler"].transform(X)
    y = BUNDLE["model"].predict(Xs)
    return y

def try_autofix_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    If uploaded CSV has no/incorrect headers but correct number of columns,
    auto-assign expected FEATURE_COLUMNS.
    """
    # Exact match → return as-is (reordered)
    if set(df.columns) == set(FEATURE_COLUMNS):
        return df[FEATURE_COLUMNS]

    # If column count matches, rename columns to expected order
    if df.shape[1] == N_FEATURES:
        fixed = df.copy()
        fixed.columns = FEATURE_COLUMNS
        return fixed

    # If there's an extra 'class' or target column, drop it
    drop_candidates = [c for c in df.columns if c.lower() in ("class", "target", "label")]
    df2 = df.drop(columns=drop_candidates, errors="ignore")
    if df2.shape[1] == N_FEATURES:
        df2 = df2.copy()
        df2.columns = FEATURE_COLUMNS
        return df2

    # No safe fix
    return None

def get_template_csv() -> bytes:
    template = pd.DataFrame(columns=FEATURE_COLUMNS)
    return template.to_csv(index=False).encode("utf-8")

# ---------- UI ----------
st.title("🧴 Skin Disorder Classification — Bulk Prediction")
st.write(
    "Upload a CSV with dermatological features to get **multi-class predictions**. "
    "The app auto-detects/repairs headers, handles numeric casting, and aligns feature order using the saved scaler."
)

with st.expander("📥 Download CSV Template (recommended)", expanded=True):
    st.write("Use this template to ensure correct columns & order.")
    st.code(", ".join(FEATURE_COLUMNS), language="text")
    st.download_button(
        label="Download template.csv",
        data=get_template_csv(),
        file_name="template_skin_disorder_features.csv",
        mime="text/csv",
        use_container_width=True,
    )

uploaded = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded is not None:
    try:
        df_raw = pd.read_csv(uploaded)
    except Exception as e:
        st.error(f"❌ Could not read CSV: {e}")
        st.stop()

    st.subheader("👀 Preview (first 5 rows)")
    st.dataframe(df_raw.head())

    # Autofix/validate columns
    df_fixed = try_autofix_columns(df_raw)

    if df_fixed is None:
        st.error(
            f"❌ Columns don't match expected features.\n\n"
            f"Expected {N_FEATURES} columns: {FEATURE_COLUMNS}\n"
            f"Uploaded: {list(df_raw.columns)}"
        )
        st.info("Tip: Download the template above or ensure your CSV has the same columns/order.")
        st.stop()

    # Numeric safety: coerce to float (turn non-numeric to NaN)
    df_num = df_fixed.apply(pd.to_numeric, errors="coerce")
    n_nan = int(df_num.isna().sum().sum())
    if n_nan > 0:
        st.warning(f"⚠️ Found {n_nan} non-numeric/missing values. They will be median-imputed per column.")
        df_num = df_num.fillna(df_num.median(numeric_only=True))

    # Action buttons
    c1, c2 = st.columns([1, 1])
    with c1:
        predict_clicked = st.button("🚀 Run Predictions", use_container_width=True)
    with c2:
        clear_clicked = st.button("🧹 Clear Upload", use_container_width=True)

    if clear_clicked:
        st.experimental_rerun()

    if predict_clicked:
        try:
            y_pred = transform_and_predict(df_num)
            out = df_raw.copy()
            out["Predicted_Class"] = y_pred.astype(int)
            st.success("✅ Predictions completed!")
            st.subheader("📄 Results (head)")
            st.dataframe(out.head())

            # Offer download
            csv_bytes = out.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="⬇️ Download predictions CSV",
                data=csv_bytes,
                file_name="skin_disorder_predictions.csv",
                mime="text/csv",
                use_container_width=True,
            )
        except Exception as e:
            st.error(f"❌ Prediction error: {e}")

# Optional: keep a minimal manual single-row predictor inside an expander
with st.expander("🧪 (Optional) Try single-row manual input"):
    cols = st.columns(4)
    row = []
    for i, feat in enumerate(FEATURE_COLUMNS):
        with cols[i % 4]:
            val = st.number_input(f"{feat}", value=0.0, step=0.1, format="%.2f")
            row.append(val)
    if st.button("Predict Single Row"):
        try:
            single = pd.DataFrame([row], columns=FEATURE_COLUMNS)
            y = transform_and_predict(single)[0]
            st.success(f"Predicted Class: {int(y)}")
        except Exception as e:
            st.error(f"❌ Error: {e}")
