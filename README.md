# CareLoop Clinics — No-Show Guard
### Predicting Outpatient Appointment No-Shows and Extracting Actionable Follow-Up Insights

[![Python 3.12](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Database: MySQL / SQLite](https://img.shields.io/badge/Database-MySQL%20%7C%20SQLite-orange.svg)](https://dev.mysql.com/)
[![GenAI: Gemini Flash](https://img.shields.io/badge/GenAI-Google%20Gemini%20Flash-8E75C2.svg)](https://ai.google.dev/)

**Repository:** [https://github.com/tarunshahi98/careloop-no-show-guard](https://github.com/tarunshahi98/careloop-no-show-guard)  
**Author:** Tarun Shahi  

---

## Key Findings

- **Data Ingestion & Cleaning:** Out of 1,020 raw booking exports, 30 invalid or missing records and 20 exact duplicates were resolved, producing 990 clean bookings with a **22.0% baseline no-show rate** (218 missed visits) and **29.9%** long-lead bookings (>14 days).
- **Spatial & Temporal Patterns (SQL):** No-show rates increase sharply with booking lead time—from **16.2%** for 1–7 days to **60.5%** for appointments booked 31+ days in advance. By clinic, **Gomti Nagar Care Point (Lucknow)** recorded the highest no-show rate at **33.3%**, followed by **Hadapsar Care Centre (Pune)** at **27.3%**.
- **Risk Modeling (ML):** A standardized Logistic Regression pipeline achieved the highest 5-fold cross-validated ROC-AUC (**0.774**) and a holdout test ROC-AUC of **0.789**. Dispatching an appointment reminder is associated with a **69% reduction in the odds of a no-show** ($\text{OR} \approx 0.31$, holding other features constant).
- **Follow-Up Note Review (GenAI):** Structured LLM classification of post-visit call-centre notes revealed that **100% (2 of 2)** of patients citing forgetfulness never received a reminder message, and **50% (5 of 10)** of all reviewed missed visits had no reminder sent.

---

## Strategic Recommendation

CareLoop should implement a two-pronged operational strategy across communications and clinic front desks:

1. **Automated Multi-Channel Reminders:** Deploy SMS and WhatsApp reminders for all bookings—especially long-lead appointments (>14 days, which account for 29.9% of visits and see no-show rates jump from 30.0% to 60.5%). Reminders are associated with a 69% decrease in no-show odds ($\text{OR} \approx 0.31$), and every patient in the sample who reported forgetting their visit had not received a reminder.
2. **Targeted Evening Outreach via Risk Scoring:** Run the Logistic Regression pipeline on the evening prior to scheduled appointments using the calibrated **0.35 risk threshold** (test ROC-AUC **0.789**, recall **57.1%**). This flags approximately **25%** of daily bookings (49 of 198 test visits) for targeted phone confirmation by front-desk staff. Prioritize high-attrition facilities first—specifically Gomti Nagar (33.3% no-show rate) and Hadapsar (27.3%)—while offering flexible time slots or teleconsultation to patients traveling longer distances.

---

## Model Evaluation & Threshold Selection

### 1. Model Comparison (5-Fold Cross-Validation on Training Set)
Models were evaluated strictly on the 792 training rows ($80\%$) using 5-fold cross-validation. The 198-row holdout test set ($20\%$) was set aside solely for final evaluation:

- **Logistic Regression Pipeline:** Mean CV ROC-AUC = **0.7737** (Selected)
- **Tuned Random Forest (`max_depth=3, n_estimators=200`):** Mean CV ROC-AUC = **0.7673**
- **Default Random Forest (`n_estimators=100`):** Mean CV ROC-AUC = **0.7466**
- **Constrained Decision Tree (`max_depth=3`):** Mean CV ROC-AUC = **0.6784**
- **Unconstrained Decision Tree:** Overfit (100% training accuracy vs 67.2% test accuracy, underperforming the 75.25% majority-class baseline).

**Why Logistic Regression was chosen:**  
Logistic Regression had the strongest cross-validation discrimination, generalized smoothly without overfitting, and produces directly interpretable odds ratios for clinical operations.

### 2. Threshold Calibration (0.50 vs 0.35)
Using out-of-fold predictions from the training set:
- **Default Threshold (0.50):** Precision = $0.5753$, Recall = $0.2485$, F1 = $0.3471$, Flagged = $73$ patients.
- **Calibrated Threshold (0.35):** Precision = $0.5130$, Recall = $0.4675$, F1 = $0.4892$, Flagged = $154$ patients.

The **0.35 threshold** nearly doubles recall ($46.8\%$ vs $24.8\%$) with only a minor drop in precision. For outpatient clinics, the cost of a missed appointment (an unfillable doctor slot and lost revenue) far outweighs the marginal cost of a short phone confirmation call.

### 3. Holdout Test Set Performance (@ 0.35 Threshold)
Evaluated once on the unseen test set ($N=198$, 49 actual no-shows):
- **Accuracy:** 0.7879
- **Precision:** 0.5714 (57.1%)
- **Recall:** 0.5714 (57.1%)
- **F1-Score:** 0.5714
- **ROC-AUC:** 0.7889
- **Flagged Appointments:** 49 of 198 (24.7%)
- **Predictions Exported:** `test_risk_scores.csv`

---

## Methodology & Limitations

1. **Observational vs. Causal Interpretation:** The dataset is observational. While reminders correlate with lower no-show odds ($\text{OR} \approx 0.31$), this does not prove direct causality; unobserved confounders (patient conscientiousness, symptom severity) may influence both reminder delivery and attendance.
2. **Preprocessing Boundary:** In Task 1.3, median imputation and IQR capping fences were calculated across the full 990 cleaned rows before train/test splitting. In strict production deployment, imputers and scalers should be fitted exclusively inside training folds.
3. **Partitioning Strategy:** A random 80/20 train/test split was used rather than a chronological split (e.g., training on earlier months and evaluating on later months). Repeat patients could also appear across both partitions.
4. **Qualitative Sample Size:** The follow-up notes analysis used the first 10 call-centre records to adhere to free-tier API quotas. While illustrative of patient reasons (e.g., transit issues, forgetting), a larger sample would be required for network-wide operational planning.

---

## Repository Structure

```
careloop-no-show-guard/
├── data/                                 # Raw data pack (read-only)
│   ├── appointments.csv                  # 1,020 raw bookings (Jan-Jun 2025)
│   ├── clinics.csv                       # 8 clinic master records
│   └── followup_notes.csv                # 30 call-centre notes
│
├── 01_ingest_clean.ipynb                 # Part 1: Ingestion, validation, cleaning, and weather API
├── 02_queries.sql                        # Part 2: Table DDL, load verification, and analytical queries
├── 03_model.ipynb                        # Part 3: ML feature prep, 5-fold CV comparison, and threshold tuning
├── 04_genai.ipynb                        # Part 4: Gemini note analysis, Pydantic validation & synthesis
│
├── appointments_clean.csv                # 990 cleaned bookings (16 columns)
├── clinics_enriched.csv                  # 8 clinics enriched with Open-Meteo weather data
├── load_data.sql                         # Bulk INSERT statements for clinics & appointments
├── notes_analysed.csv                    # Structured note insights (10 records)
├── rejected_rows.json                    # 32 rejected records from raw CSV parsing
├── test_risk_scores.csv                  # Holdout test set predictions (198 rows, risk_prob & flag)
│
├── src/                                  # Modular Python source code
│   ├── __init__.py
│   ├── data_ingest.py                    # Ingestion, cleaning, and IQR capping
│   ├── db_loader.py                      # SQLite database ingestion utilities
│   ├── model_pipeline.py                 # Scikit-learn pipeline and threshold predictor
│   └── genai_service.py                  # Pydantic schema and Gemini service
│
├── .gitignore
└── README.md
```

---

## How to Run

### 1. Environment Setup
```bash
# Clone the repository
git clone https://github.com/tarunshahi98/careloop-no-show-guard.git
cd careloop-no-show-guard

# Create and activate virtual environment
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate

# Install dependencies
pip install pandas numpy scikit-learn pydantic requests google-genai
```

### 2. SQL Database Setup
In MySQL 8.0 Workbench or SQLiteOnline:
1. Run table creation statements from `02_queries.sql`.
2. Execute `load_data.sql` to populate clinics and clean appointments.
3. Run the analytical queries in `02_queries.sql`.

### 3. Notebook Execution
Run the notebooks in order:
- `01_ingest_clean.ipynb`: Cleans data and generates `appointments_clean.csv`, `clinics_enriched.csv`, and `load_data.sql`.
- `03_model.ipynb`: Performs 5-fold CV model selection, tunes threshold to 0.35, and outputs `test_risk_scores.csv`.
- `04_genai.ipynb`: Analyzes call notes with Gemini and Pydantic validation, generating `notes_analysed.csv`.

---

## Environment & Security
- **No Credentials in Repo:** All API keys and secrets are loaded via environment variables or runtime input; no keys or credentials exist in version control.
- **Reproducibility:** All results, metrics, and tables can be reproduced directly from the raw data files.
