# DataLens

> **Clean your data. Understand your numbers.**

DataLens is a lightweight, transparent data cleaning and insight workspace built with **Python**, **Pandas**, and **Streamlit**. It transforms messy, inconsistent CSV files into clean, analysis-ready datasets with zero black-box magic, zero external databases, and zero fake AI gimmicks.

---

## Table of Contents

1. [Product Overview](#product-overview)
2. [Key Features](#key-features)
3. [Technology Stack](#technology-stack)
4. [Quickstart & Installation](#quickstart--installation)
5. [User Workflow](#user-workflow)
6. [Architecture & Code Structure](#architecture--code-structure)
7. [Signature Feature: The Cleaning Receipt](#signature-feature-the-cleaning-receipt)
8. [Automated Testing & Verification](#automated-testing--verification)
9. [Interview Questions & Detailed Answers](#interview-questions--detailed-answers)
10. [Known Limitations & Scope](#known-limitations--scope)

---

## Product Overview

Many data tools are either bloated enterprise software or fragile student scripts. DataLens fills the gap as a polished, thoughtful independent workspace designed for analysts, researchers, and developers who need **complete transparency** over their data preparation.

### Core Value Proposition
- **Full Transparency:** See exactly what is wrong, preview proposed changes, approve each modification, and receive a permanent audit receipt.
- **Safety by Design:** Raw data is stored immutably in memory. No destructive operation ever alters the original uploaded data silently. You can reset anytime.
- **Factual Analytics:** Insights and visualizations are derived strictly from descriptive statistics (like Tukey’s 1.5×IQR dispersion rule), never fabricated AI claims.

---

## Key Features

### 1. Data Inspection & Health Profiling
- **Dimensions:** Instant calculation of row, column, and total cell counts.
- **Duplicates:** Full-row duplicate detection with immediate preview.
- **Missingness Audit:** Total missing values and per-column missing percentage breakdown.
- **Blank String Detection:** Identifies misleading empty or whitespace-only cells (`"   "`) masquerading as valid text.
- **Column Typing:** Distinguishes Numeric, Categorical/Text, Datetime, and Boolean columns with 3-value non-null sample previews.
- **Cleanliness Score:** Real-time health index measuring non-missing, non-blank valid cells.

### 2. Interactive Cleaning Workspace
- **Remove Duplicate Rows:** Identifies exact duplicates, presents an inspection table, and drops duplicates with one-click confirmation.
- **Trim Text Whitespace:** Identifies cells with accidental leading/trailing spaces (e.g., `'  Electronics '` $\to$ `'Electronics'`) across selectable text columns.
- **Blank Strings to NaN:** Converts empty and whitespace-only text values into formal `NaN` values for uniform missing-data handling.
- **Capitalization Normalization:** Standardizes erratic casing (e.g., `'north'`, `'NORTH'`, and `'North'`) to **Title Case**, **UPPERCASE**, or **lowercase** only when explicitly chosen.
- **Missing Value Strategies:**
  - *Leave unchanged.*
  - *Remove affected rows* containing missing values in target columns.
  - *Fill numeric missing values* with column median (calculated dynamically).
  - *Fill text missing values* with a custom user-defined replacement (e.g., `"Unknown"` or `"Unspecified"`).
- **Type Conversion:** Offers safe conversion of text columns to Numeric or Datetime. Previews parseable percentages and reports unparseable values converted to `NaN`.

### 3. Signature Feature: The Cleaning Receipt
- Automatically logs every single applied operation chronologically.
- Tracks operation name, target column(s), timestamp, number of rows or values affected, and a plain-English explanation.
- Displays side-by-side **Before vs. After** metrics (rows, columns, missing values, duplicates).
- Downloadable as a formatted **Text Audit Report (`.txt`)** and a **Structured Audit CSV (`.csv`)**.

### 4. Automatic Visualizations ("Explore Your Data")
- **Summary Statistics:** Mean, median, standard deviation, minimum, 25th percentile, 75th percentile, and maximum for numeric columns.
- **Numeric Distribution:** Dynamic 12-bin histogram frequency chart using Streamlit's native visualization engine.
- **Categorical Frequencies:** Top N frequency bar chart with count labels.
- **Time-Series Trends:** Observations over time and metric aggregation across detected Datetime fields.
- **Two-Variable Scatter Plot:** Multi-variable relationship exploration for numeric pairs.

### 5. Factual Statistical Insights
- **Completeness Evaluation:** Highlights the column with the highest missingness rate.
- **Dominant Category Detection:** Identifies categorical concentration (e.g., `'Technology'` accounts for 35.5% of rows).
- **Dispersion & Statistical Outlier Flags:** Uses Tukey's 1.5×IQR boundary rule ($Q_1 - 1.5 \times IQR$ and $Q_3 + 1.5 \times IQR$) to flag unusual extremes for human review.
- **Unique Identifier Detection:** Identifies columns with 100% unique values suitable as primary keys.

---

## Technology Stack

| Technology | Role |
| :--- | :--- |
| **Python 3.10+** | Core programming language & standard libraries (`io`, `os`, `datetime`, `math`, `unittest`) |
| **Pandas** | Tabular data ingestion, vector operations, profiling, and cleaning transformations |
| **NumPy** | Statistical computations, array operations, and quantile calculations |
| **Streamlit** | Interactive SaaS UI, reactive state management, tables, built-in charts, and file downloads |

*Zero external database, cloud service, AI model, or JavaScript framework required.*

---

## Quickstart & Installation

### 1. Clone or Navigate to the Repository
```bash
cd ExpensePredictor
```

### 2. Create and Activate a Virtual Environment
```bash
# On Windows
python -m venv .venv
.\.venv\Scripts\activate

# On macOS/Linux
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Launch DataLens
```bash
streamlit run app.py
```
Your browser will open automatically at `http://localhost:8501`.

---

## User Workflow

```
┌─────────────────┐     ┌──────────────────────┐     ┌────────────────────────┐
│  1. Ingestion   │ ──> │ 2. Quality Audit     │ ──> │ 3. Clean Workspace     │
│ Upload CSV /    │     │ Review rows, cols,   │     │ Preview & apply        │
│ Load Sample Data│     │ duplicates & missing │     │ controlled operations  │
└─────────────────┘     └──────────────────────┘     └────────────────────────┘
                                                                  │
                                                                  ▼
┌─────────────────┐     ┌──────────────────────┐     ┌────────────────────────┐
│   6. Export     │ <── │  5. Visualizations   │ <── │ 4. Cleaning Receipt    │
│ Download CSV &  │     │ Distributions, time  │     │ Real-time diffs &      │
│ Audit Receipts  │     │ series, scatter, mode│     │ chronological ledger   │
└─────────────────┘     └──────────────────────┘     └────────────────────────┘
```

1. **Open DataLens:** The user lands on a clean, minimal workspace with navigation on the left.
2. **Load Dataset:** Click *"Load Sample B2B Sales Dataset"* or upload any CSV file (up to 50MB).
3. **Inspect Raw Data:** Review the high-level metrics cards, dataset preview, and column breakdown.
4. **Clean Step-by-Step:**
   - Review proposed changes with live previews before execution.
   - Confirm duplicate removal, strip whitespace, handle missing values, or normalize text casing.
   - Watch the active Cleanliness Score update in real time.
5. **Explore Visualizations & Insights:** Switch to *"Explore Your Data"* or *"Factual Insights"* to review statistical trends and outliers.
6. **Download Cleaned Outputs:** Navigate to *"Export & Receipt"* to download the cleaned CSV alongside the `.txt` or `.csv` audit receipt.
7. **Reset Anytime:** Click *"↺ Reset to Raw Data"* in the sidebar to revert all transformations and return to the baseline data.

---

## Architecture & Code Structure

The project strictly adheres to separation of concerns. Core calculations and transformations are isolated into a pure Python module (`data_processor.py`), completely decoupled from the Streamlit UI layer (`app.py`).

```
ExpensePredictor/
├── .streamlit/
│   └── config.toml           # Streamlit theme configuration (indigo & neutral tones)
├── sample_data.csv           # Realistic fictional B2B sales dataset with intentional flaws
├── test_inventory.csv        # Secondary warehouse inventory dataset for verification
├── data_processor.py         # Pure functional data processing, profiling, and receipt engine
├── app.py                    # Streamlit presentation, reactive UI, and visualization layouts
├── test_datalens.py          # Standard unittest test suite
├── verify_end_to_end.py      # End-to-end integration and edge-case verification script
├── requirements.txt          # Minimal dependency manifest (streamlit, pandas, numpy)
├── .gitignore                # Git exclusions (virtualenvs, cache, OS files)
└── README.md                 # Product documentation and interview guide
```

### Module Responsibilities

#### `data_processor.py` (Core Engine)
- `load_data()`: Validates file size, checks non-emptiness, attempts fallback encodings (`utf-8`, `latin1`, `cp1252`), and catches malformed delimiters.
- `profile_data()`: Calculates health metrics without mutating the DataFrame.
- `preview_*()` and `apply_*()`: Paired functions for each cleaning action ensuring that users can inspect what will happen before committing a change.
- `CleaningReceiptTracker`: An audit class tracking chronological operations, diff calculations, and receipt text/CSV generation.
- `generate_insights()`: Computes objective statistical facts (highest missingness, mode, unique identifiers, and IQR-based statistical flags).

#### `app.py` (Presentation & State Management)
- Maintains session state (`raw_df`, `clean_df`, `receipt_tracker`, `data_source_name`).
- Renders custom SaaS CSS with accessible contrasts, clean typography, and restrained metric cards.
- Manages navigation across Overview, Clean Data, Explore Data, Insights, and Export.
- Handles reactive UI triggers, column multi-selects, preview tables, and download buttons.

---

## Signature Feature: The Cleaning Receipt

DataLens treats reproducibility and accountability as first-class citizens. The **Cleaning Receipt** tracks every operation in memory:

```text
============================================================
                   DATALENS CLEANING RECEIPT                 
         Clean your data. Understand your numbers.          
============================================================
Generated At       : 2026-11-03 20:32:58
Operations Applied : 7
------------------------------------------------------------
DATASET SUMMARY COMPARISON
------------------------------------------------------------
Rows               : 35 -> 32 (Diff: -3)
Columns            : 10 -> 10
Missing Values     : 6 -> 0 (Diff: -6)
Duplicate Rows     : 3 -> 0
------------------------------------------------------------
CHRONOLOGICAL LOG OF OPERATIONS
------------------------------------------------------------
Step 1 [20:32:58] Remove Duplicate Rows
  Target Column(s) : All Columns
  Affected Count   : 3
  Explanation      : Removed 3 exact duplicate row(s). Row count changed from 35 to 32.

Step 2 [20:32:58] Trim Whitespace
  Target Column(s) : customer_name, unit_price, payment_status
  Affected Count   : 5
  Explanation      : Trimmed leading and trailing whitespace from 5 cell(s) across 3 column(s).
...
============================================================
End of Cleaning Receipt. Preserved by DataLens.
============================================================
```

Both `.txt` and `.csv` formats are available for instant export.

---

## Automated Testing & Verification

DataLens includes two independent test suites:

### 1. Unit Tests (`test_datalens.py`)
Tests every individual function with isolated synthetic test cases:
```bash
python test_datalens.py
```
*Covers: valid and invalid loading, profiling counts, duplicate drop, whitespace trimming, blank string conversion, casing normalization, median imputation, custom text fill, numeric conversion, date conversion, receipt summaries, and statistical insights.*

### 2. End-to-End Verification Suite (`verify_end_to_end.py`)
Simulates the entire user journey on the included business sample dataset, warehouse inventory dataset, and edge-case inputs:
```bash
python verify_end_to_end.py
```

---

## Interview Questions & Detailed Answers

### Q1: How does DataLens ensure data safety and prevent accidental data loss?
**Answer:**  
DataLens enforces immutability on the uploaded dataset. When a file is loaded, it is saved into `st.session_state["raw_df"]` and never modified. All cleaning operations create copies of the active DataFrame (`st.session_state["clean_df"]`). Furthermore, every destructive action (such as dropping rows or coercing invalid types) features a **pre-execution preview** showing the exact rows or counts that will be affected. If a user makes an error, the *"↺ Reset to Raw Data"* action instantly restores the original dataset and clears the receipt log.

### Q2: Why is separating pure logic (`data_processor.py`) from the UI layer (`app.py`) an important architectural choice?
**Answer:**  
Streamlit's execution model re-runs the entire script upon user interaction. Mixing data manipulation logic directly with UI widgets makes unit testing difficult, complicates debugging, and couples business logic to Streamlit's runtime. By placing all data loading, profiling, cleaning, receipt generation, and insight algorithms in `data_processor.py`, the core engine is 100% testable via standard Python tools (`unittest`, `pytest`) without spinning up a browser or mocking Streamlit sessions.

### Q3: How does DataLens distinguish between a statistical outlier and an erroneous data point?
**Answer:**  
DataLens strictly avoids claiming that unusual numbers are errors. It applies **Tukey’s standard IQR dispersion rule** ($Q_1 - 1.5 \times IQR$ to $Q_3 + 1.5 \times IQR$). When a value lies beyond these thresholds, DataLens displays it as an objective **Statistical Flag** for analyst review, with an explicit disclaimer: *"This is an objective statistical flag based on dispersion, not proof that the record is incorrect."* This prevents users from blindly deleting valid high-value transactions or edge cases.

### Q4: How does DataLens manage memory and state in a multi-step Streamlit workflow without a database?
**Answer:**  
DataLens leverages Streamlit's built-in session state (`st.session_state`). Because DataLens is scoped for single-user analytical sessions, maintaining references to the raw DataFrame, cleaned DataFrame, and the `CleaningReceiptTracker` instance in memory eliminates the latency, security vulnerabilities, and deployment overhead of an external database. State persists seamlessly across widget interactions within the user's active session.

### Q5: Why was Tukey's IQR method chosen for statistical outlier detection over z-scores or machine learning models?
**Answer:**  
The z-score method relies on the sample mean and standard deviation, both of which are notoriously sensitive to the very extreme values they attempt to detect. Machine learning models (like Isolation Forests) introduce unnecessary black-box dependencies and hyperparameter tuning. Tukey's IQR relies on quartiles (order statistics), making it robust to extreme skews while remaining mathematically transparent and easy for any stakeholder to interpret.

---

## Known Limitations & Scope

In accordance with scope discipline, DataLens intentionally avoids unnecessary bloat:
1. **Memory Ceiling:** Processing runs in-memory; datasets larger than 100MB may experience UI slowdown depending on client RAM. A safe 50MB upload limit is enforced by default.
2. **File Formats:** Only tabular delimiter-separated CSV files are supported. Hierarchical formats (JSON trees, XML) and multi-sheet Excel files (`.xlsx`) are not accepted.
3. **Single-Session Scope:** Cleaning receipts and datasets reside in transient session state. Refreshing or closing the browser clears the session. Permanent records must be downloaded using the provided export buttons.
