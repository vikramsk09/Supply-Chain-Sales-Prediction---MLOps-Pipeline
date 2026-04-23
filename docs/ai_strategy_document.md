# AI Strategy Document — Team C
## Supply Chain Analytics & Sales Prediction System
### Evoastra Ventures (OPC) Pvt Ltd — March 2026

---

## Executive Summary

This document consolidates the end-to-end findings, methodologies, and strategic outcomes from all three phases of the Evoastra Supply Chain Analytics Capstone. The system was designed and deployed to transform raw operational data into intelligent, explainable, and production-ready machine learning predictions.

The project progressed through three distinct analytical layers:

| Phase | Focus | Outcome |
|-------|-------|---------|
| Phase 1 | Data Analytics & EDA | Cleaned dataset, KPIs, operational insights |
| Phase 2 | Predictive Modeling & Forecasting | Regression models, time-series, feature intelligence |
| Phase 3 | Advanced ML & MLOps Pipeline | XGBoost (R² = 0.967), SHAP explainability, CI/CD deployment |

**Core system transformation:**
> Descriptive Analytics → Predictive Analytics → Intelligent Decision Support → Production Deployment

---

## 1. Problem Statement

Modern supply chains generate massive operational data but often lack the analytical infrastructure to convert that data into decisions. This project addresses four core business problems:

- **Demand Forecasting** — Predict future sales values to enable inventory pre-positioning
- **Delay Risk Identification** — Identify orders at high risk of late delivery before they occur
- **Profitability Analysis** — Understand which products, regions, and customers drive or erode margins
- **Decision Support** — Provide explainable, stakeholder-readable predictions via SHAP and FastAPI

**Dataset:** DataCo Global Supply Chain — 180,519 rows, 53 columns, covering order details, logistics, product attributes, customer data, and financial outcomes.

---

## 2. Phase 1 — Data Analytics & Exploratory Intelligence

### 2.1 Objective

Ingest, clean, and explore the supply chain dataset to uncover operational patterns, define KPIs, identify inefficiencies, and prepare data for modeling.

### 2.2 Data Cleaning & Preparation

The raw dataset required significant cleaning before it was analytically usable:

- **Dropped columns:** `Product Description` (99.7% missing) and `Order Zipcode` (86% missing)
- **Removed PII columns:** `customer_email`, `customer_password`, `customer_fname/lname`, `customer_street`, `latitude`, `longitude`, `product_image`
- **Standardized column names:** Stripped whitespace, converted to lowercase snake_case
- **Converted date columns** to `datetime64` format to enable temporal analysis
- **Removed highly correlated features** (correlation > 0.9) to reduce multicollinearity — including `customer_id`, `sales_per_customer`, `benefit_per_order`, `order_item_id`
- **Noise reduction:** Standardized all categorical text columns (lowercase, strip spaces)
- **Final dataset:** 180,508 rows × 40 columns (after all cleaning)
- **Zero duplicate records** confirmed

### 2.3 Feature Engineering (Phase 1)

| Feature | Description |
|---------|-------------|
| `delivery_days` | `shipping_date − order_date` (actual delivery duration) |
| `delay_flag` | Binary: 1 = Late delivery, 0 = On-time |
| `order_day`, `order_week`, `order_month` | Extracted from `order_date` |
| `is_weekend` | Whether the order was placed on a weekend |
| `sales_log` | Log transform of sales (skewness: 2.88 → −0.68) |
| `profit_log` | Flipped log transform of profit (skewness: −4.74 → −0.47) |

### 2.4 Key Findings from EDA

**Delivery Performance**

| Delivery Status | Percentage |
|-----------------|------------|
| Late delivery | 54.83% |
| Advance shipping | 23.04% |
| On time | 17.83% |
| Cancelled | 4.29% |

Over half of all orders are delivered late. However, the average delay is only ~0.56 days (median: 1 day), indicating the problem is systemic small delays rather than catastrophic failures. Delivery delays are consistent across departments and regions, confirming this is an operational issue, not a localized one.

**Product & Category Insights**

Sports-related products dominate: Cleats (24,551 orders), Men's Footwear (22,246), Women's Apparel (21,035) are the top three categories. Most profitable categories are Boxing & MMA, Baseball & Softball, and Cleats. Low-margin categories include Toys, Books, and CDs.

**Financial Performance**

| KPI | Value |
|-----|-------|
| Total Sales | $36.78 million |
| Average Profit per Order | $21.97 |
| Total Profit | $3.96 million |
| Average Discount Rate | 10.16% |
| Average Delivery Days | 3.47 days |

The sales-profit correlation is weak (~0.13), meaning high sales do not guarantee profitability. Negative profit orders (minimum: −$4,274) exist even at high sales values, driven by excessive discounting or high shipping costs.

**Geographic Concentration**

Primary markets are the United States (111,137 orders) and Puerto Rico (69,371 orders). Revenue is heavily concentrated in Caguas, followed by Chicago, Los Angeles, and New York — indicating both strength and geographic risk.

### 2.5 Business Recommendations (Phase 1)

- Improve logistics scheduling to reduce the system-wide ~0.56 day delay
- Optimize inventory distribution closer to high-demand cities
- Review discount policies to eliminate negative-profit transactions
- Prioritize high-margin sports equipment categories in marketing and inventory allocation
- Implement predictive delay models (addressed in Phases 2 & 3)

---

## 3. Phase 2 — Predictive Modeling & Statistical Validation

### 3.1 Objective

Transform the cleaned dataset into a predictive intelligence layer — forecast demand, validate business hypotheses statistically, and identify key feature drivers.

> Phase shift: **Descriptive analytics (what happened) → Predictive analytics (what will happen)**

### 3.2 Statistical Hypothesis Testing

| Test | Result | Business Insight |
|------|--------|-----------------|
| T-Test (payment mode vs delivery time) | p > 0.05 — not significant | Binary payment comparison insufficient to explain delays |
| ANOVA (multi-category delivery comparison) | p ≪ 0.05 — highly significant | Specific transaction categories significantly affect delivery time |
| Chi-Square (payment type vs delivery status) | p ≈ 0 — strong dependency | Certain payment modes introduce systemic processing delays |

### 3.3 Feature Engineering (Phase 2)

Building on Phase 1 features, Phase 2 added temporal intelligence:

| Feature | Purpose |
|---------|---------|
| `lag_1` | Previous period demand — captures demand memory |
| `rolling_mean` (7-day) | Short-term demand smoothing |
| `order_day`, `order_week`, `order_month`, `weekday` | Seasonality and weekly patterns |

**Key Insight:** Demand in supply chains is not independent — it is correlated with recent historical behavior. Lag and rolling features were critical for model performance.

### 3.4 Regression Models

Three models were trained and evaluated on the same feature set:

| Model | RMSE | MAE | R² |
|-------|------|-----|----|
| Linear Regression | ~35 | ~18 | ~0.91 |
| Ridge Regression | ~35 | ~18 | ~0.91 |
| Lasso Regression | ~35.08 | ~18.23 | 0.9306 |

All three models performed similarly, confirming that the data exhibits strong linear characteristics. Lasso was selected as the Phase 2 best model due to its feature selection properties and marginally better R². 5-fold cross-validation confirmed stability with consistent RMSE across folds.

### 3.5 Time Series Forecasting

- **ARIMA:** Captures trend in demand; suitable for short-term forecasting; does not model seasonality
- **SARIMA:** Extends ARIMA with seasonal components; more accurate and realistic; confirms supply chain demand exhibits trend, seasonality, and temporal correlation

### 3.6 Feature Importance (Mutual Information)

Top influential features identified in Phase 2:

1. `product_price` — price elasticity effect
2. `lag_1` — previous demand memory
3. `rolling_mean` — short-term demand trend
4. `order_item_quantity` — volume driver
5. `order_month` — seasonality signal

### 3.7 Correlation Analysis

- **Pearson:** Strong correlation between Sales ↔ Price and Sales ↔ Quantity; moderate between delivery time and shipping time
- **Spearman:** Reveals additional non-linear temporal patterns not captured by Pearson — motivating the need for advanced ensemble models in Phase 3

### 3.8 Operational Insights (Phase 2)

- Payment-related delays exist and are statistically significant
- Demand variability requires adaptive inventory planning
- Linear models hit a performance ceiling — non-linear ensemble methods needed
- Feature leakage and label encoding issues identified for correction in Phase 3

---

## 4. Phase 3 — Advanced ML, Explainability & MLOps Deployment

### 4.1 Objective

Advance from predictive modeling to an intelligent, production-grade pipeline with ensemble ML, cross-validation, hyperparameter tuning, SHAP explainability, and CI/CD deployment.

> Phase shift: **Predictive System (Phase 2) → Intelligent Decision System (Phase 3)**

### 4.2 System Architecture

```
DataCoSupplyChain.csv
    → Preprocessing (preprocess.py)
        → Feature Alignment (columns.pkl)
            → ColumnTransformer (OneHotEncoder + Numerical Scaling)
                → XGBoost Model
                    → Prediction (log scale → expm1)
                        → SHAP Explanation
                            → FastAPI Response
```

**CI/CD Pipeline:**
```
Code Push / Schedule
    → GitHub Actions
        → train.py (retrain model)
        → predict.py (test inference)
        → Update artifacts/
            → Auto Deploy (Railway / Render)
                → FastAPI Response
```

### 4.3 Models Trained

**13 input features:** shipping/delivery days, order quantity, date components, and 7 label-encoded categoricals (shipping_mode, order_status, market, customer_segment, order_region, category_name, department_name). Features scaled via StandardScaler.

**Random Forest (initial):**
- n_estimators = 100, max_depth = None, random_state = 42
- Ensemble of decision trees; reduces variance through averaging
- Captures non-linear interactions that linear Phase 2 models missed

**XGBoost (initial):**
- n_estimators = 100, learning_rate = 0.1, random_state = 42
- Sequential gradient boosting; each tree corrects errors of the previous
- Superior handling of mixed feature types with built-in regularization

### 4.4 Cross-Validation

3-fold KFold cross-validation (shuffle=True) applied via `cross_val_predict`:

| Model | CV R² | CV SAE | Interpretation |
|-------|--------|--------|----------------|
| Random Forest | 0.964 | Lower | Stable across all 3 folds |
| XGBoost | 0.967 | Lowest | Best generalization, minimal overfitting |

Cross-validation confirms both models generalize well. XGBoost outperforms on every metric.

### 4.5 Hyperparameter Tuning (GridSearchCV)

**Random Forest:**

| Parameter | Values Searched | Best |
|-----------|----------------|------|
| n_estimators | 50, 100 | 100 |
| max_depth | 5, 10 | 10 |
| min_samples_split | 2, 5 | 2 |

**XGBoost:**

| Parameter | Values Searched | Best |
|-----------|----------------|------|
| n_estimators | 50, 100 | 100 |
| learning_rate | 0.05, 0.1 | 0.1 |
| max_depth | 3, 5 | 5 |

Tuning improved both models; XGBoost showed the larger gain.

### 4.6 Final Model Evaluation

| Phase | Model | R² | RMSE | MAE |
|-------|-------|----|------|-----|
| Phase 2 (Baseline) | Lasso Regression | 0.9306 | 35.08 | 18.23 |
| Phase 3 | Random Forest (Tuned) | 0.964 | 22.14 | — |
| **Phase 3** | **XGBoost (Tuned) ★** | **0.967** | **20.87** | — |

**40% reduction in RMSE** from Phase 2 baseline to Phase 3 best model. Both ensemble models exceed the ≥15% improvement threshold over baseline.

### 4.7 SHAP Explainability

SHAP (SHapley Additive exPlanations) was applied to the tuned XGBoost model for global and local interpretation:

| Plot Type | Scope | What It Shows |
|-----------|-------|---------------|
| Summary Plot | Global | SHAP value distribution across all features and predictions |
| Bar Plot | Global | Mean absolute SHAP per feature — overall importance ranking |
| Force Plot | Local | Per-prediction feature contributions |
| Dependence Plot | Global | How a feature's value affects its SHAP impact |

**Top SHAP Drivers:**

| Rank | Feature | Direction | Business Meaning |
|------|---------|-----------|-----------------|
| 1 | `days_for_shipping_(real)` | Strongest driver | Faster delivery → higher predicted sales |
| 2 | `delivery_days` | Very High | Logistics speed directly impacts revenue |
| 3 | `order_item_quantity` | High positive | Volume consistently drives higher predictions |
| 4 | `product_price` | High | Price elasticity captured — high-price items shift predictions significantly |
| 5 | `lag_1` | Moderate-High | Previous demand memory validated as significant |
| 6 | `rolling_mean` | Moderate | 7-day demand smoothing contributes meaningfully |
| 7–13 | Encoded categoricals | Lower | Market, region, segment signals — weaker than logistics features |

**Critical Insight:** Logistics features dominate. Reducing shipping days is the single highest-impact operational lever for improving supply chain revenue performance.

### 4.8 Example Prediction (Single Order)

**Input:**
- Type: DEBIT
- Days for shipping (real): 3
- Product Price: $327.75
- Customer Segment: Consumer

**Pipeline:**
```python
# Preprocessing → Feature Alignment → Encoding
df = df.reindex(columns=columns, fill_value=0)
X_transformed = preprocessor.transform(df_single)

# Prediction (reverse log transform)
pred_log = model.predict(X_transformed)
prediction = np.expm1(pred_log)
```

**Output: $327.80**

**SHAP Attribution for this prediction:**
- `Sales Per Customer` → +0.72 (increases prediction)
- `Department: Fitness` → −0.04 (decreases prediction)
- `Category: Sporting Goods` → −0.009 (decreases prediction)

---

## 5. Unified Feature Intelligence

Across all three phases, the following features consistently emerged as the most influential:

| Feature | Phase 1 Signal | Phase 2 Mutual Info | Phase 3 SHAP Rank |
|---------|---------------|--------------------|--------------------|
| `days_for_shipping_(real)` | Top delay driver | High | #1 |
| `delivery_days` | Correlated with shipping | High | #2 |
| `order_item_quantity` | Volume pattern | #4 | #3 |
| `product_price` | Profit analysis | #1 | #4 |
| `lag_1` | Not yet present | #2 | #5 |
| `rolling_mean` | Not yet present | #3 | #6 |

Logistics features consistently outrank product, customer, and geographic features across all measurement methods — validating that **operational efficiency is the primary revenue lever** in this supply chain system.

---

## 6. Strategic Business Impact

| Business Area | Finding | Recommended Action |
|--------------|---------|-------------------|
| **Demand Planning** | R² = 0.967 enables reliable 7–30 day demand forecasts | Use model for inventory pre-positioning |
| **Logistics Optimization** | Shipping speed is the #1 revenue driver (SHAP) | Prioritize fast-lane carriers for high-value orders |
| **Delivery Delays** | 54.83% late delivery rate; avg delay 0.56 days | Improve scheduling system-wide, not regionally |
| **Pricing Strategy** | Product price SHAP confirms price elasticity | Support dynamic pricing decisions with model outputs |
| **Profitability** | High sales ≠ high profit; discount rate negatively correlated with profit | Review discount policies; protect margins on high-volume SKUs |
| **Geographic Strategy** | Revenue concentrated in Caguas and major US cities | Diversify to reduce concentration risk |
| **Category Strategy** | Boxing & MMA, Baseball, Cleats = highest margin | Increase marketing and inventory allocation to these categories |
| **Decision Support** | SHAP force plots explain individual predictions | Empower non-technical managers to act on model outputs |
| **Model Governance** | Cross-validation + SHAP = auditable, explainable AI pipeline | Ready for enterprise deployment and regulatory review |

---

## 7. Known Limitations & Risk Scenarios

| Risk | Description | Mitigation |
|------|-------------|-----------|
| **Demand Spikes** | No anomaly detection for black swan events | Add Isolation Forest or statistical control charts |
| **Cold Start** | New products/customers not modeled | Implement content-based fallback or Bayesian prior |
| **Concept Drift** | Model accuracy degrades as supply chain patterns shift | Implement Evidently AI monitoring; set retraining triggers |
| **External Shocks** | Geopolitical disruptions, weather, macroeconomic shifts not modeled | Integrate external signals as features in future iterations |
| **Real-Time Gap** | Current system is batch prediction only | Phase 4 objective: Kafka/Spark streaming pipeline |
| **Label Encoding** | High-cardinality categoricals may lose ordinal meaning | Replace with Target Encoding in next model version |

---

## 8. Optimization Roadmap

### Immediate (Next Sprint)
- Wrap preprocessing + model into a single `sklearn.Pipeline` object for reproducibility
- Serialize `best_xgb` with MLflow for version-controlled model registry
- Implement Evidently AI for prediction drift and data drift monitoring

### Short-Term (1–3 Months)
- Add LightGBM as a third ensemble candidate for speed-accuracy benchmarking
- Apply Bayesian optimization (Optuna) to replace GridSearchCV for faster tuning
- Replace Label Encoding with Target Encoding for high-cardinality categoricals
- Add supplier performance scores, inventory levels, and lead time variability as features

### Medium-Term (3–6 Months)
- Move from batch inference to real-time scoring via Kafka + Spark
- Explore stacking/blending RF and XGBoost predictions for marginal accuracy gains
- Integrate external signals: promotional calendars, regional economic indicators
- Build anomaly detection layer for demand spike identification

---

## 9. Technology Stack

| Layer | Tools |
|-------|-------|
| Data Processing | Python, Pandas, NumPy |
| Analytics & Visualization | Matplotlib, Seaborn, Plotly, Power BI |
| Statistical Analysis | SciPy, Statsmodels |
| Machine Learning | Scikit-learn, XGBoost, Random Forest |
| Time Series | ARIMA, SARIMA, Prophet |
| Explainability | SHAP (summary, force, dependence plots) |
| Hyperparameter Tuning | GridSearchCV, Optuna |
| Deployment | FastAPI, Docker, docker-compose |
| MLOps | MLflow, GitHub Actions |
| Monitoring | Evidently AI, Grafana |
| Hosting | Railway / Render |

---

## 10. Final Conclusion

This AI strategy document captures the full journey of the Evoastra Supply Chain Analytics system — from raw, uncleaned operational data to a production-grade, explainable, and deployable machine learning pipeline.

**Key Achievements Across All Phases:**

- Cleaned and structured a 180K+ row real-world supply chain dataset
- Identified that 54.83% of orders are delayed — a systemic logistics issue requiring operational intervention
- Delivered a regression baseline (Lasso, R² = 0.9306) and advanced it to XGBoost (R² = 0.967) — a 40% reduction in RMSE
- Validated model performance through 3-fold and 5-fold cross-validation across all phases
- Confirmed with SHAP that model decisions are driven by logistics-relevant signals, not data artefacts
- Deployed a CI/CD-backed FastAPI inference endpoint with automated retraining

**The system's strategic value lies not only in its predictive accuracy, but in its explainability.** Every prediction can be traced to specific business drivers — giving supply chain managers the confidence to act on model outputs rather than treat them as black boxes.

---

*© 2026 Evoastra Ventures (OPC) Pvt Ltd. All rights reserved.*
*Document prepared by the Supply Chain Analytics Team C — March 2026*
