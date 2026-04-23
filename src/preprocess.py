import pandas as pd
import numpy as np

def preprocess(df):
    df = df.copy()

    # -------------------------
    # STANDARDIZE COLUMN NAMES
    # -------------------------
    df.columns = df.columns.str.lower().str.replace(" ", "_")

    # -------------------------
    # HANDLE DATE COLUMN
    # -------------------------
    if 'shipping_date_(dateorders)' in df.columns:
        df['shipping_date'] = pd.to_datetime(
            df['shipping_date_(dateorders)'], errors='coerce'
        )
    else:
        df['shipping_date'] = pd.NaT

    # -------------------------
    # HANDLE SALES SAFELY
    # -------------------------
    if 'sales' not in df.columns:
        df['sales'] = 0

    # -------------------------
    # HANDLE PROFIT SAFELY
    # -------------------------
    if 'order_profit_per_order' in df.columns:
        df['profit'] = df['order_profit_per_order']
    else:
        df['profit'] = 0

    # -------------------------
    # FEATURE ENGINEERING
    # -------------------------
    if 'days_for_shipping_(real)' in df.columns:
        df['delivery_days'] = df['days_for_shipping_(real)']
    else:
        df['delivery_days'] = 0

    df['order_month'] = df['shipping_date'].dt.month.fillna(0)
    df['order_day'] = df['shipping_date'].dt.day.fillna(0)
    df['order_week'] = df['shipping_date'].dt.isocalendar().week.astype(float).fillna(0)
    df['is_weekend'] = df['shipping_date'].dt.weekday.fillna(0) >= 5

    if 'delivery_status' in df.columns:
        df['delay_flag'] = (df['delivery_status'] == 'late delivery').astype(int)
    else:
        df['delay_flag'] = 0

    # -------------------------
    # SAFE LOG TRANSFORM
    # -------------------------
    df['sales'] = pd.to_numeric(df['sales'], errors='coerce').fillna(0).clip(lower=0)
    df['profit'] = pd.to_numeric(df['profit'], errors='coerce').fillna(0).clip(lower=0)

    df['sales_log'] = np.log1p(df['sales'])
    df['profit_log'] = np.log1p(df['profit'])

    # -------------------------
    # DROP USELESS / LEAKAGE / PROBLEMATIC COLUMNS
    # -------------------------
    drop_cols = [
        'shipping_date',
        'shipping_date_(dateorders)',
        'order_date_(dateorders)',

        'order_id',
        'order_item_id',
        'customer_id',
        'product_card_id',

        'customer_email',
        'customer_password',
        'product_description',
        'product_image',

        'sales',
        'profit'
    ]

    df = df.drop(columns=drop_cols, errors='ignore')

    # -------------------------
    # FORCE NUMERIC CONSISTENCY
    # -------------------------
    num_cols = df.select_dtypes(include=['int64', 'float64']).columns
    for col in num_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    # -------------------------
    # FORCE CATEGORICAL CONSISTENCY
    # -------------------------
    cat_cols = df.select_dtypes(include=['object', 'category']).columns
    for col in cat_cols:
        df[col] = df[col].astype(str)

    return df