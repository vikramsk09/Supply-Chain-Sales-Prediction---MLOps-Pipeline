import pandas as pd
import numpy as np
import joblib
from preprocess import preprocess

# Load artifacts
model = joblib.load("artifacts/model.pkl")
columns = joblib.load("artifacts/columns.pkl")

def predict_from_csv(path):
    # -------------------------
    # LOAD DATA (FIX ENCODING)
    # -------------------------
    df = pd.read_csv(path, encoding="latin1")

    # -------------------------
    # PREPROCESS
    # -------------------------
    df = preprocess(df)

    # -------------------------
    # ALIGN COLUMNS
    # -------------------------
    df = df.reindex(columns=columns, fill_value=0)

    # -------------------------
    # PREDICT (LOG SCALE)
    # -------------------------
    preds_log = model.predict(df)

    # -------------------------
    # CONVERT BACK TO REAL SALES
    # -------------------------
    preds = np.expm1(preds_log)

    return preds


if __name__ == "__main__":
    preds = predict_from_csv("DataCoSupplyChain.csv")

    print("Sample Predictions (Real Sales):")
    print(preds[:5])