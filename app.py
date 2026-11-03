"""
DataLens - A Lightweight Data Cleaning and Insight Workspace
Tagline: Clean your data. Understand your numbers.

Built with Python, Pandas, and Streamlit.
Zero external database, AI, or separate backend.
"""

from datetime import datetime
import io
import os
import pandas as pd
import numpy as np
import streamlit as st

from data_processor import (
    load_data,
    profile_data,
    infer_column_type,
    preview_remove_duplicates,
    apply_remove_duplicates,
    preview_whitespace_trim,
    apply_whitespace_trim,
    preview_blank_strings,
    apply_blank_strings_to_nan,
    preview_casing_normalization,
    apply_casing_normalization,
    preview_missing_handling,
    apply_missing_handling,
    preview_type_conversion,
    apply_type_conversion,
    CleaningReceiptTracker,
    generate_insights,
)

# =====================================================================
# Page Configuration & SaaS-Grade Minimalist Styling
# =====================================================================

st.set_page_config(
    page_title="DataLens | Clean your data. Understand your numbers.",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

CUSTOM_CSS = """
<style>
    /* Global Reset & Base Typography */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
        color: #1e293b;
    }

    /* Main Container Padding */
    .block-container {
        padding-top: 1.75rem;
        padding-bottom: 2.5rem;
        padding-left: 2.5rem;
        padding-right: 2.5rem;
        max-width: 1280px;
    }

    /* Header & Branding */
    .brand-title {
        font-size: 1.6rem;
        font-weight: 700;
        letter-spacing: -0.025em;
        color: #0f172a;
        margin-bottom: 0.2rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    .brand-tagline {
        font-size: 0.92rem;
        font-weight: 500;
        color: #64748b;
        margin-bottom: 1.5rem;
    }
    .brand-pill {
        display: inline-block;
        background-color: #e0e7ff;
        color: #3730a3;
        font-size: 0.72rem;
        font-weight: 600;
        padding: 0.2rem 0.55rem;
        border-radius: 9999px;
        letter-spacing: 0.04em;
        text-transform: uppercase;
    }

    /* SaaS Metric Cards */
    .metric-card {
        background-color: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 1rem 1.15rem;
        box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.03);
        margin-bottom: 0.85rem;
    }
    .metric-title {
        font-size: 0.76rem;
        font-weight: 600;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 0.35rem;
    }
    .metric-value {
        font-size: 1.55rem;
        font-weight: 700;
        color: #0f172a;
        line-height: 1.2;
    }
    .metric-delta-good {
        font-size: 0.78rem;
        color: #059669;
        font-weight: 600;
        margin-top: 0.25rem;
    }
    .metric-delta-warn {
        font-size: 0.78rem;
        color: #d97706;
        font-weight: 600;
        margin-top: 0.25rem;
    }

    /* Insight Banner Cards */
    .insight-card {
        background-color: #ffffff;
        border-left: 4px solid #4f46e5;
        border-top: 1px solid #f1f5f9;
        border-right: 1px solid #f1f5f9;
        border-bottom: 1px solid #f1f5f9;
        border-radius: 0 8px 8px 0;
        padding: 1rem 1.25rem;
        margin-bottom: 0.85rem;
        box-shadow: 0 1px 2px rgba(0,0,0,0.03);
    }
    .insight-badge {
        font-size: 0.7rem;
        font-weight: 600;
        padding: 0.15rem 0.45rem;
        border-radius: 4px;
        text-transform: uppercase;
        letter-spacing: 0.03em;
        display: inline-block;
        margin-bottom: 0.4rem;
    }
    .badge-primary { background-color: #ede9fe; color: #5b21b6; }
    .badge-success { background-color: #dcfce7; color: #15803d; }
    .badge-warning { background-color: #fef3c7; color: #b45309; }
    .badge-info { background-color: #e0f2fe; color: #0369a1; }

    .insight-title {
        font-size: 0.95rem;
        font-weight: 600;
        color: #1e293b;
        margin-bottom: 0.25rem;
    }
    .insight-detail {
        font-size: 0.84rem;
        color: #475569;
        line-height: 1.45;
    }

    /* Receipt Card */
    .receipt-container {
        background-color: #fafafa;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 1.25rem;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.82rem;
        color: #334155;
    }

    /* Table styling tweaks */
    div[data-testid="stDataFrame"] {
        border-radius: 8px;
        overflow: hidden;
        border: 1px solid #e2e8f0;
    }

    /* Clean Buttons */
    .stButton > button {
        border-radius: 6px;
        font-weight: 500;
        font-size: 0.86rem;
        transition: all 0.15s ease-in-out;
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# =====================================================================
# State Initialization
# =====================================================================

def init_session_state():
    """Initializes Streamlit session state keys."""
    if "raw_df" not in st.session_state:
        st.session_state["raw_df"] = None
    if "clean_df" not in st.session_state:
        st.session_state["clean_df"] = None
    if "data_source_name" not in st.session_state:
        st.session_state["data_source_name"] = None
    if "receipt_tracker" not in st.session_state:
        st.session_state["receipt_tracker"] = None
    if "active_nav" not in st.session_state:
        st.session_state["active_nav"] = "Overview"


init_session_state()


def set_dataset(df: pd.DataFrame, source_name: str):
    """Sets a new active dataset, preserving the raw copy and initializing the receipt tracker."""
    st.session_state["raw_df"] = df.copy()
    st.session_state["clean_df"] = df.copy()
    st.session_state["data_source_name"] = source_name
    st.session_state["receipt_tracker"] = CleaningReceiptTracker(df)


def reset_to_raw():
    """Resets the cleaned dataset back to the untouched raw state."""
    if st.session_state["raw_df"] is not None:
        st.session_state["clean_df"] = st.session_state["raw_df"].copy()
        st.session_state["receipt_tracker"] = CleaningReceiptTracker(st.session_state["raw_df"])
        st.toast("Restored raw dataset. Cleaning receipt reset.", icon="🔄")


# =====================================================================
# Sidebar: Navigation, Upload & Dataset Controls
# =====================================================================

with st.sidebar:
    st.markdown(
        """
        <div style="padding-top: 0.5rem; margin-bottom: 1.25rem;">
            <div style="font-size: 1.35rem; font-weight: 700; color: #1e1b4b; display: flex; align-items: center; gap: 0.4rem;">
                <span style="color: #4f46e5;">◈</span> DataLens
            </div>
            <div style="font-size: 0.78rem; font-weight: 500; color: #64748b; margin-top: 0.15rem;">
                Clean your data. Understand your numbers.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("### Workspace Navigation")
    nav_options = [
        "Overview",
        "Clean Your Data",
        "Explore Your Data",
        "Factual Insights",
        "Export & Receipt",
    ]
    selected_nav = st.radio(
        label="Navigation Menu",
        options=nav_options,
        index=nav_options.index(st.session_state["active_nav"]) if st.session_state["active_nav"] in nav_options else 0,
        label_visibility="collapsed",
    )
    st.session_state["active_nav"] = selected_nav

    st.markdown("---")

    # Data Source Section
    st.markdown("### Load Dataset")
    uploaded_file = st.file_uploader(
        "Upload CSV (Max 50MB)",
        type=["csv"],
        help="Upload any standard comma-separated tabular CSV file.",
    )

    if uploaded_file is not None:
        # Load uploaded file
        loaded_df, err = load_data(uploaded_file, max_size_mb=50.0)
        if err:
            st.error(err)
        else:
            if (
                st.session_state["raw_df"] is None
                or st.session_state["data_source_name"] != uploaded_file.name
            ):
                set_dataset(loaded_df, uploaded_file.name)
                st.success(f"Loaded: {uploaded_file.name}")

    # Sample Dataset Button
    st.markdown(
        "<div style='font-size: 0.8rem; color: #64748b; margin-top: 0.4rem;'>Or start with realistic fictional business data:</div>",
        unsafe_allow_html=True,
    )

    sample_path = "sample_data.csv"
    if st.button("Load Sample B2B Sales Dataset", use_container_width=True):
        if os.path.exists(sample_path):
            sample_df, err = load_data(sample_path)
            if err:
                st.error(err)
            else:
                set_dataset(sample_df, "Sample B2B Sales Dataset (sample_data.csv)")
                st.toast("Sample B2B Sales dataset loaded successfully!", icon="📊")
        else:
            st.error("Sample dataset file not found on disk.")

    # Status & Reset Badge
    if st.session_state["clean_df"] is not None:
        st.markdown("---")
        st.markdown("### Dataset Status")
        st.caption(f"**Source:** {st.session_state['data_source_name']}")
        ops_applied = len(st.session_state["receipt_tracker"].operations) if st.session_state["receipt_tracker"] else 0
        st.caption(f"**Operations applied:** {ops_applied}")

        if st.button("↺ Reset to Raw Data", use_container_width=True, help="Reverts all cleaning actions and restores the raw uploaded file."):
            reset_to_raw()
            st.rerun()


# =====================================================================
# Main Application Content
# =====================================================================

def render_empty_state():
    """Renders a polite, helpful empty state when no dataset is loaded."""
    st.markdown(
        """
        <div style="background-color: #ffffff; border: 2px dashed #cbd5e1; border-radius: 12px; padding: 3rem 2rem; text-align: center; margin-top: 1.5rem;">
            <div style="font-size: 2.2rem; margin-bottom: 0.75rem;">📊</div>
            <div style="font-size: 1.25rem; font-weight: 700; color: #0f172a; margin-bottom: 0.4rem;">
                No Dataset Loaded Yet
            </div>
            <div style="font-size: 0.9rem; color: #64748b; max-width: 520px; margin: 0 auto 1.5rem auto; line-height: 1.5;">
                Upload your CSV file using the sidebar on the left, or click the button below to explore DataLens with our pre-loaded business sample dataset.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    col1, col2, col3 = st.columns([1, 1.2, 1])
    with col2:
        if st.button("📁 Load Sample Business Dataset", use_container_width=True, type="primary"):
            if os.path.exists("sample_data.csv"):
                sample_df, err = load_data("sample_data.csv")
                if not err:
                    set_dataset(sample_df, "Sample B2B Sales Dataset (sample_data.csv)")
                    st.rerun()


# Guard: Check if data is loaded
if st.session_state["clean_df"] is None:
    render_empty_state()
    st.stop()

# Data is active
current_df: pd.DataFrame = st.session_state["clean_df"]
raw_df: pd.DataFrame = st.session_state["raw_df"]
tracker: CleaningReceiptTracker = st.session_state["receipt_tracker"]
profile = profile_data(current_df)
raw_profile = profile_data(raw_df)


# =====================================================================
# Section 1: Overview & Data Inspection
# =====================================================================

if st.session_state["active_nav"] == "Overview":
    st.markdown(
        """
        <div class="brand-title">
            <span>Overview & Data Quality Inspection</span>
            <span class="brand-pill">Audit Mode</span>
        </div>
        <div class="brand-tagline">
            Examine your dataset's structural dimensions, missingness, duplicate rows, and column types.
        </div>
        """,
        unsafe_allow_html=True,
    )

    # 4 Quick Metric Cards
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">Total Rows</div>
                <div class="metric-value">{profile['total_rows']:,}</div>
                <div class="metric-delta-good">Active Records</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">Columns</div>
                <div class="metric-value">{profile['total_columns']:,}</div>
                <div style="font-size: 0.78rem; color: #64748b; font-weight: 500; margin-top: 0.25rem;">Dimensions</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col3:
        dup_style = "metric-delta-warn" if profile['duplicate_rows'] > 0 else "metric-delta-good"
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">Duplicate Rows</div>
                <div class="metric-value">{profile['duplicate_rows']}</div>
                <div class="{dup_style}">{'Needs Action' if profile['duplicate_rows'] > 0 else 'Clean'}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col4:
        miss_style = "metric-delta-warn" if profile['total_missing'] > 0 else "metric-delta-good"
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">Missing Values</div>
                <div class="metric-value">{profile['total_missing']}</div>
                <div class="{miss_style}">{profile['missing_pct']}% of all cells</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col5:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">Cleanliness Score</div>
                <div class="metric-value">{profile['cleanliness_score']}%</div>
                <div class="metric-delta-good">Data Health</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Issue Alert Box if issues exist
    issues_detected = []
    if profile["duplicate_rows"] > 0:
        issues_detected.append(f"**{profile['duplicate_rows']} exact duplicate row(s)** detected.")
    if profile["total_missing"] > 0:
        issues_detected.append(f"**{profile['total_missing']} missing cell(s)** across the dataset.")
    if profile["blank_strings_total"] > 0:
        issues_detected.append(f"**{profile['blank_strings_total']} blank/whitespace-only string(s)** should be treated as NaN.")

    if issues_detected:
        st.warning(
            "⚠️ **Data Quality Issues Detected:** " + " ".join(issues_detected) +
            " Head over to **'Clean Your Data'** to review and resolve them."
        )
    else:
        st.success("✅ **Clean Baseline:** No duplicate rows or blank strings detected in the current view.")

    # Data Preview
    st.markdown("### Dataset Preview")
    preview_slice = st.slider("Preview Rows", min_value=5, max_value=min(100, len(current_df)), value=min(15, len(current_df)))
    st.dataframe(current_df.head(preview_slice), use_container_width=True)

    # Detailed Column Health Table
    st.markdown("### Column Data Types & Quality Breakdown")
    st.dataframe(profile["column_stats"], use_container_width=True, hide_index=True)


# =====================================================================
# Section 2: Cleaning Workspace
# =====================================================================

elif st.session_state["active_nav"] == "Clean Your Data":
    st.markdown(
        """
        <div class="brand-title">
            <span>Clean Your Data</span>
            <span class="brand-pill">Controlled Operations</span>
        </div>
        <div class="brand-tagline">
            Inspect proposed changes with transparency before applying. The original raw data remains safely preserved.
        </div>
        """,
        unsafe_allow_html=True,
    )

    clean_tabs = st.tabs([
        "1. Duplicate Rows",
        "2. Text Whitespace",
        "3. Blank Strings to NaN",
        "4. Capitalization",
        "5. Missing Values",
        "6. Type Conversion",
    ])

    # -------------------------------------------------------------
    # Tab 1: Duplicate Rows
    # -------------------------------------------------------------
    with clean_tabs[0]:
        st.markdown("#### Remove Exact Duplicate Rows")
        st.markdown(
            "Detects and removes rows where every single column value matches an existing preceding row."
        )

        dup_preview = preview_remove_duplicates(current_df)
        if dup_preview["duplicate_count"] == 0:
            st.info("✓ No exact duplicate rows found in the current dataset.")
        else:
            st.warning(
                f"⚠️ Found **{dup_preview['duplicate_count']} duplicate row(s)**. "
                "Review the preview below before confirming removal."
            )
            st.markdown("**Duplicate Rows Preview:**")
            st.dataframe(dup_preview["preview_df"], use_container_width=True)

            col_btn, _ = st.columns([1, 3])
            with col_btn:
                if st.button("Confirm & Remove Duplicates", type="primary", key="btn_remove_duplicates"):
                    cleaned, receipt = apply_remove_duplicates(current_df)
                    st.session_state["clean_df"] = cleaned
                    tracker.record(receipt)
                    st.success(receipt["explanation"])
                    st.rerun()

    # -------------------------------------------------------------
    # Tab 2: Text Whitespace Trimming
    # -------------------------------------------------------------
    with clean_tabs[1]:
        st.markdown("#### Trim Leading & Trailing Text Whitespace")
        st.markdown(
            "Removes accidental leading and trailing whitespace from string values (e.g., `'  Electronics '` → `'Electronics'`)."
        )

        ws_preview = preview_whitespace_trim(current_df)
        if ws_preview["affected_cells"] == 0:
            st.info("✓ No untrimmed whitespace detected in text columns.")
        else:
            st.warning(
                f"⚠️ Detected **{ws_preview['affected_cells']} cell(s)** with leading or trailing whitespace "
                f"across {len(ws_preview['affected_columns'])} column(s): `{', '.join(ws_preview['affected_columns'])}`."
            )

            # Option to choose specific columns or all
            target_cols = st.multiselect(
                "Columns to Trim:",
                options=ws_preview["target_columns"],
                default=ws_preview["affected_columns"],
                key="ws_selected_columns"
            )

            if not ws_preview["samples_df"].empty:
                st.markdown("**Sample Values Needing Trimming:**")
                st.dataframe(ws_preview["samples_df"], use_container_width=True, hide_index=True)

            col_btn, _ = st.columns([1, 3])
            with col_btn:
                if st.button("Apply Whitespace Trimming", type="primary", key="btn_trim_ws"):
                    if not target_cols:
                        st.error("Please select at least one column to trim.")
                    else:
                        cleaned, receipt = apply_whitespace_trim(current_df, target_cols)
                        st.session_state["clean_df"] = cleaned
                        tracker.record(receipt)
                        st.success(receipt["explanation"])
                        st.rerun()

    # -------------------------------------------------------------
    # Tab 3: Blank Strings to NaN
    # -------------------------------------------------------------
    with clean_tabs[2]:
        st.markdown("#### Detect Blank Strings as Missing (NaN)")
        st.markdown(
            "Whitespace-only strings (e.g. `'   '`) or empty strings (`''`) can deceptively disguise themselves as valid text. "
            "Converting them to formal `NaN` allows standard missing-value handling to treat them properly."
        )

        bs_preview = preview_blank_strings(current_df)
        if bs_preview["affected_cells"] == 0:
            st.info("✓ No blank or whitespace-only text cells detected.")
        else:
            st.warning(
                f"⚠️ Detected **{bs_preview['affected_cells']} blank string(s)** in columns: "
                f"`{', '.join(bs_preview['affected_columns'])}`."
            )

            if not bs_preview["samples_df"].empty:
                st.markdown("**Sample Blank Cells:**")
                st.dataframe(bs_preview["samples_df"], use_container_width=True, hide_index=True)

            target_cols_bs = st.multiselect(
                "Columns to Convert to NaN:",
                options=bs_preview["target_columns"],
                default=bs_preview["affected_columns"],
                key="bs_selected_columns"
            )

            col_btn, _ = st.columns([1, 3])
            with col_btn:
                if st.button("Convert Blank Strings to NaN", type="primary", key="btn_apply_blank_nan"):
                    if not target_cols_bs:
                        st.error("Please select at least one column.")
                    else:
                        cleaned, receipt = apply_blank_strings_to_nan(current_df, target_cols_bs)
                        st.session_state["clean_df"] = cleaned
                        tracker.record(receipt)
                        st.success(receipt["explanation"])
                        st.rerun()

    # -------------------------------------------------------------
    # Tab 4: Capitalization Normalization
    # -------------------------------------------------------------
    with clean_tabs[3]:
        st.markdown("#### Normalize Inconsistent Text Capitalization")
        st.markdown(
            "Standardizes inconsistent casing (e.g. `'north'`, `'NORTH'`, and `'North'` into a consistent format). "
            "Applied **only** when explicitly selected."
        )

        text_cols = [c for c in current_df.columns if current_df[c].dtype == "object" or pd.api.types.is_string_dtype(current_df[c])]
        if not text_cols:
            st.info("No text columns available in the current dataset.")
        else:
            col_sel1, col_sel2 = st.columns(2)
            with col_sel1:
                selected_casing_cols = st.multiselect(
                    "Select Column(s) to Normalize:",
                    options=text_cols,
                    key="casing_target_cols",
                )
            with col_sel2:
                selected_style = st.selectbox(
                    "Standardization Style:",
                    options=["Title Case", "UPPERCASE", "lowercase"],
                    key="casing_style_choice",
                )

            if selected_casing_cols:
                preview_casing = preview_casing_normalization(current_df, selected_casing_cols, selected_style)
                st.caption(f"Estimated **{preview_casing['affected_cells']} cell(s)** will change casing.")

                if not preview_casing["samples_df"].empty:
                    st.markdown("**Proposed Changes Preview:**")
                    st.dataframe(preview_casing["samples_df"], use_container_width=True, hide_index=True)

                col_btn, _ = st.columns([1, 3])
                with col_btn:
                    if st.button(f"Apply {selected_style}", type="primary", key="btn_apply_casing"):
                        cleaned, receipt = apply_casing_normalization(
                            current_df, selected_casing_cols, selected_style
                        )
                        st.session_state["clean_df"] = cleaned
                        tracker.record(receipt)
                        st.success(receipt["explanation"])
                        st.rerun()

    # -------------------------------------------------------------
    # Tab 5: Handle Missing Values
    # -------------------------------------------------------------
    with clean_tabs[4]:
        st.markdown("#### Handle Missing Values")
        st.markdown(
            "Choose a targeted missing value strategy for selected columns. Never silently deletes or invents data."
        )

        cols_with_missing = [c for c in current_df.columns if current_df[c].isna().sum() > 0]

        if not cols_with_missing:
            st.success("✓ There are zero missing values in the current dataset.")
        else:
            st.warning(
                f"Columns with missing values: " +
                ", ".join([f"**{c}** ({current_df[c].isna().sum()})" for c in cols_with_missing])
            )

            col_m1, col_m2 = st.columns(2)
            with col_m1:
                selected_missing_cols = st.multiselect(
                    "Target Column(s):",
                    options=current_df.columns.tolist(),
                    default=cols_with_missing,
                    key="missing_target_cols",
                )
            with col_m2:
                missing_strategy = st.selectbox(
                    "Handling Strategy:",
                    options=[
                        "Leave unchanged",
                        "Remove affected rows",
                        "Fill numeric with column median",
                        "Fill text with custom value",
                    ],
                    key="missing_strategy_choice",
                )

            custom_text_val = "Unknown"
            if missing_strategy == "Fill text with custom value":
                custom_text_val = st.text_input("Custom Fill Value for Text:", value="Unknown")

            if selected_missing_cols:
                m_preview = preview_missing_handling(
                    current_df, selected_missing_cols, missing_strategy, custom_text_val
                )
                st.info(f"📋 **Action Preview:** {m_preview['message']}")

                col_btn, _ = st.columns([1, 3])
                with col_btn:
                    if st.button("Apply Missing Value Treatment", type="primary", key="btn_apply_missing"):
                        if missing_strategy == "Leave unchanged":
                            st.info("Strategy is set to 'Leave unchanged'. No modifications applied.")
                        else:
                            cleaned, receipt = apply_missing_handling(
                                current_df, selected_missing_cols, missing_strategy, custom_text_val
                            )
                            st.session_state["clean_df"] = cleaned
                            tracker.record(receipt)
                            st.success(receipt["explanation"])
                            st.rerun()

    # -------------------------------------------------------------
    # Tab 6: Type Conversion (Numeric & Datetime)
    # -------------------------------------------------------------
    with clean_tabs[5]:
        st.markdown("#### Numeric & Date Type Conversion")
        st.markdown(
            "Convert text columns containing numbers or dates into proper data types. "
            "Invalid conversions are reported transparently without corrupting valid data."
        )

        candidate_cols = current_df.columns.tolist()
        col_c1, col_c2 = st.columns(2)
        with col_c1:
            target_conv_col = st.selectbox("Select Column to Convert:", options=candidate_cols, key="conv_col_choice")
        with col_c2:
            target_type = st.selectbox("Target Data Type:", options=["Numeric", "Datetime"], key="conv_type_choice")

        if target_conv_col:
            conv_preview = preview_type_conversion(current_df, target_conv_col, target_type)
            st.info(f"📋 **Conversion Feasibility:** {conv_preview['message']}")

            if conv_preview["invalid_count"] > 0:
                st.warning(
                    f"⚠️ {conv_preview['invalid_count']} value(s) cannot be parsed as {target_type} "
                    f"and will become `NaN`. Unparseable samples: `{conv_preview['invalid_sample']}`"
                )

            col_btn, _ = st.columns([1, 3])
            with col_btn:
                if st.button(f"Convert '{target_conv_col}' to {target_type}", type="primary", key="btn_apply_conv"):
                    cleaned, receipt = apply_type_conversion(current_df, target_conv_col, target_type)
                    st.session_state["clean_df"] = cleaned
                    tracker.record(receipt)
                    st.success(receipt["explanation"])
                    st.rerun()

    # -------------------------------------------------------------
    # Cleaning Receipt Display (Signature Feature)
    # -------------------------------------------------------------
    st.markdown("---")
    st.markdown("### 🧾 Cleaning Receipt Audit Trail")
    st.caption("A transparent chronological ledger of all modifications applied to the active dataset.")

    rec_summary = tracker.get_summary(current_df)
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Initial vs Cleaned Rows", f"{rec_summary['cleaned_rows']:,}", delta=f"{rec_summary['rows_diff']:+d}")
    with c2:
        st.metric("Initial vs Cleaned Cols", f"{rec_summary['cleaned_cols']:,}")
    with c3:
        st.metric("Missing Values", f"{rec_summary['cleaned_missing']:,}", delta=f"{rec_summary['missing_diff']:+d}", delta_color="inverse")
    with c4:
        st.metric("Duplicate Rows", f"{rec_summary['cleaned_duplicates']:,}")

    if not tracker.operations:
        st.info("No cleaning operations have been performed yet. As you clean, each step will be permanently recorded here.")
    else:
        ops_df = pd.DataFrame(tracker.operations)
        st.dataframe(ops_df, use_container_width=True, hide_index=True)


# =====================================================================
# Section 3: Automatic Visualizations
# =====================================================================

elif st.session_state["active_nav"] == "Explore Your Data":
    st.markdown(
        """
        <div class="brand-title">
            <span>Explore Your Data</span>
            <span class="brand-pill">Visualizations</span>
        </div>
        <div class="brand-tagline">
            Inspect column distributions, categories, and patterns using Streamlit's native charts.
        </div>
        """,
        unsafe_allow_html=True,
    )

    numeric_cols = [c for c in current_df.columns if pd.api.types.is_numeric_dtype(current_df[c])]
    categorical_cols = [c for c in current_df.columns if current_df[c].dtype == "object" or pd.api.types.is_string_dtype(current_df[c])]
    datetime_cols = [c for c in current_df.columns if pd.api.types.is_datetime64_any_dtype(current_df[c])]

    chart_tabs = st.tabs([
        "Summary Statistics",
        "Numeric Distribution",
        "Categorical Frequencies",
        "Time-Series Trends",
        "Two-Variable Scatter",
    ])

    # -------------------------------------------------------------
    # Tab 1: Summary Statistics
    # -------------------------------------------------------------
    with chart_tabs[0]:
        st.markdown("#### Numeric Summary Statistics")
        if not numeric_cols:
            st.info("No numeric columns detected in the dataset.")
        else:
            stats_df = current_df[numeric_cols].describe().T
            stats_df["median"] = current_df[numeric_cols].median()
            # Rearrange columns nicely
            cols_order = ["count", "mean", "median", "std", "min", "25%", "50%", "75%", "max"]
            valid_order = [c for c in cols_order if c in stats_df.columns]
            st.dataframe(stats_df[valid_order].round(2), use_container_width=True)

    # -------------------------------------------------------------
    # Tab 2: Numeric Distribution (Histogram / Bins)
    # -------------------------------------------------------------
    with chart_tabs[1]:
        st.markdown("#### Numeric Distribution")
        if not numeric_cols:
            st.info("No numeric columns available to plot.")
        else:
            selected_num = st.selectbox("Select Numeric Column:", options=numeric_cols, key="viz_num_col")
            s = current_df[selected_num].dropna()

            if s.empty:
                st.warning("Selected column contains only missing values.")
            else:
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Min", f"{s.min():.2f}")
                c2.metric("Median", f"{s.median():.2f}")
                c3.metric("Mean", f"{s.mean():.2f}")
                c4.metric("Max", f"{s.max():.2f}")

                # Bin distribution into 10 intervals
                try:
                    counts, bin_edges = np.histogram(s, bins=12)
                    bin_labels = [f"{bin_edges[i]:.1f} - {bin_edges[i+1]:.1f}" for i in range(len(counts))]
                    dist_df = pd.DataFrame({"Count": counts}, index=bin_labels)
                    st.bar_chart(dist_df, use_container_width=True)
                    st.caption(f"Frequency distribution for **'{selected_num}'** partitioned into 12 equal-width bins.")
                except Exception as ex:
                    st.error(f"Could not render distribution: {str(ex)}")

    # -------------------------------------------------------------
    # Tab 3: Categorical Frequencies
    # -------------------------------------------------------------
    with chart_tabs[2]:
        st.markdown("#### Categorical Frequency Distribution")
        if not categorical_cols:
            st.info("No text or categorical columns available to plot.")
        else:
            selected_cat = st.selectbox("Select Categorical Column:", options=categorical_cols, key="viz_cat_col")
            top_n = st.slider("Top N Values to Display:", min_value=5, max_value=25, value=10)

            val_counts = current_df[selected_cat].dropna().value_counts().head(top_n)
            if val_counts.empty:
                st.warning("Selected column has no valid values to chart.")
            else:
                st.bar_chart(val_counts, use_container_width=True)
                st.caption(f"Top {len(val_counts)} most frequent entries in **'{selected_cat}'**.")

    # -------------------------------------------------------------
    # Tab 4: Time-Series Trends
    # -------------------------------------------------------------
    with chart_tabs[3]:
        st.markdown("#### Observations Over Time")
        if not datetime_cols:
            st.info(
                "💡 **No datetime columns detected.** "
                "If your dataset includes a date column (such as `'order_date'`), you can convert it to Datetime "
                "in the **'Clean Your Data' > 'Type Conversion'** tab to unlock time-series visualizations."
            )
        else:
            col_d1, col_d2 = st.columns(2)
            with col_d1:
                selected_dt = st.selectbox("Select Datetime Column:", options=datetime_cols, key="viz_dt_col")
            with col_d2:
                metric_options = ["Observation Frequency (Row Count)"] + numeric_cols
                selected_metric = st.selectbox("Metric to Plot Over Time:", options=metric_options, key="viz_dt_metric")

            ts_df = current_df.dropna(subset=[selected_dt]).copy()
            ts_df["_date_period"] = ts_df[selected_dt].dt.to_period("D").dt.to_timestamp()

            if selected_metric == "Observation Frequency (Row Count)":
                ts_chart_data = ts_df.groupby("_date_period").size().rename("Observations")
            else:
                ts_chart_data = ts_df.groupby("_date_period")[selected_metric].sum()

            st.line_chart(ts_chart_data, use_container_width=True)
            st.caption(f"Timeline visualization plotted against **'{selected_dt}'**.")

    # -------------------------------------------------------------
    # Tab 5: Two-Variable Scatter Plot
    # -------------------------------------------------------------
    with chart_tabs[4]:
        st.markdown("#### Two-Variable Scatter Plot")
        if len(numeric_cols) < 2:
            st.info("At least two numeric columns are required to generate a scatter plot.")
        else:
            col_x, col_y = st.columns(2)
            with col_x:
                x_col = st.selectbox("Select X-Axis Variable:", options=numeric_cols, index=0, key="viz_scatter_x")
            with col_y:
                y_idx = 1 if len(numeric_cols) > 1 else 0
                y_col = st.selectbox("Select Y-Axis Variable:", options=numeric_cols, index=y_idx, key="viz_scatter_y")

            scatter_data = current_df[[x_col, y_col]].dropna()
            if scatter_data.empty:
                st.warning("No rows have non-null values for both selected variables.")
            else:
                st.scatter_chart(scatter_data, x=x_col, y=y_col, use_container_width=True)
                st.caption(f"Scatter plot comparing **'{x_col}'** (horizontal) against **'{y_col}'** (vertical).")


# =====================================================================
# Section 4: Factual Insights
# =====================================================================

elif st.session_state["active_nav"] == "Factual Insights":
    st.markdown(
        """
        <div class="brand-title">
            <span>Factual Insights</span>
            <span class="brand-pill">Statistical Observations</span>
        </div>
        <div class="brand-tagline">
            Objective mathematical observations derived directly from your numbers. No hallucinated AI claims.
        </div>
        """,
        unsafe_allow_html=True,
    )

    insights = generate_insights(current_df, raw_df)

    badge_class_map = {
        "Missing Values": "badge-warning",
        "100% Complete": "badge-success",
        "Dominant Category": "badge-info",
        "Statistical Flag": "badge-warning",
        "Unique Identifier": "badge-success",
        "Data Quality Boost": "badge-success",
    }

    for ins in insights:
        badge_cls = badge_class_map.get(ins.get("badge", ""), "badge-primary")
        border_color = "#10b981" if ins.get("type") == "success" else ("#f59e0b" if ins.get("type") == "warning" else "#4f46e5")

        st.markdown(
            f"""
            <div class="insight-card" style="border-left-color: {border_color};">
                <span class="insight-badge {badge_cls}">{ins.get('badge', 'Insight')}</span>
                <div class="insight-title">{ins.get('title', '')}</div>
                <div class="insight-detail">{ins.get('detail', '')}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown(
        """
        <div style="background-color: #f1f5f9; border-radius: 6px; padding: 0.85rem 1.15rem; font-size: 0.78rem; color: #64748b; margin-top: 1.5rem;">
            ℹ️ <strong>Statistical Integrity Note:</strong> All findings above are calculated using standard descriptive statistics 
            (including Tukey's 1.5×IQR dispersion rule and mode calculations). Any extreme values flagged are statistical indicators 
            intended for human review, not proof that a transaction or record is erroneous.
        </div>
        """,
        unsafe_allow_html=True,
    )


# =====================================================================
# Section 5: Export & Receipt
# =====================================================================

elif st.session_state["active_nav"] == "Export & Receipt":
    st.markdown(
        """
        <div class="brand-title">
            <span>Export Cleaned Data & Receipt</span>
            <span class="brand-pill">Audit Ready</span>
        </div>
        <div class="brand-tagline">
            Download your finalized, cleaned dataset alongside complete audit documentation.
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Summary metrics
    rec_summary = tracker.get_summary(current_df)
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">Cleaned Rows</div>
                <div class="metric-value">{rec_summary['cleaned_rows']:,}</div>
                <div class="metric-delta-good">Original: {rec_summary['initial_rows']:,}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">Remaining Missing Cells</div>
                <div class="metric-value">{rec_summary['cleaned_missing']}</div>
                <div class="metric-delta-good">Original: {rec_summary['initial_missing']}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">Audit Trail Steps</div>
                <div class="metric-value">{rec_summary['operations_count']}</div>
                <div style="font-size: 0.78rem; color: #64748b; margin-top: 0.25rem;">Operations Applied</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Download Buttons
    st.markdown("### Download Deliverables")
    down_col1, down_col2, down_col3 = st.columns(3)

    # 1. Cleaned CSV
    cleaned_csv_bytes = current_df.to_csv(index=False).encode("utf-8")
    with down_col1:
        st.download_button(
            label="📥 Download Cleaned CSV",
            data=cleaned_csv_bytes,
            file_name="datalens_cleaned_data.csv",
            mime="text/csv",
            use_container_width=True,
            type="primary",
        )

    # 2. Text Audit Receipt
    receipt_txt = tracker.generate_text_receipt(current_df)
    with down_col2:
        st.download_button(
            label="📄 Download Receipt (TXT)",
            data=receipt_txt,
            file_name="datalens_cleaning_receipt.txt",
            mime="text/plain",
            use_container_width=True,
        )

    # 3. CSV Audit Log
    receipt_csv = tracker.generate_csv_receipt()
    with down_col3:
        st.download_button(
            label="📊 Download Receipt (CSV)",
            data=receipt_csv,
            file_name="datalens_cleaning_receipt.csv",
            mime="text/csv",
            use_container_width=True,
        )

    # Receipt Preview Container
    st.markdown("### Receipt Preview")
    st.text_area(
        "Audit Log Text Representation",
        value=receipt_txt,
        height=320,
        disabled=True,
        label_visibility="collapsed",
    )
