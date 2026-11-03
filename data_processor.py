"""
data_processor.py - Pure Data Processing Engine for DataLens
Provides transparent data loading, profiling, cleaning, receipt generation, and statistical insights.
Uses only Python standard library and Pandas.
"""

from datetime import datetime
import io
import math
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd


def load_data(
    file_or_path: Union[str, io.BytesIO, Any],
    max_size_mb: float = 50.0
) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
    """
    Safely loads a CSV file into a Pandas DataFrame.
    Validates file size, non-emptiness, and handles syntax/encoding errors gracefully.

    Returns:
        (DataFrame, None) on success
        (None, error_message) on failure
    """
    # 1. Size validation
    try:
        if isinstance(file_or_path, str):
            import os
            if not os.path.exists(file_or_path):
                return None, f"File not found: {file_or_path}"
            size_bytes = os.path.getsize(file_or_path)
        elif hasattr(file_or_path, "size"):
            size_bytes = file_or_path.size
        elif hasattr(file_or_path, "getbuffer"):
            size_bytes = file_or_path.getbuffer().nbytes
        elif hasattr(file_or_path, "seek") and hasattr(file_or_path, "tell"):
            file_or_path.seek(0, io.SEEK_END)
            size_bytes = file_or_path.tell()
            file_or_path.seek(0)
        else:
            size_bytes = 0

        max_bytes = max_size_mb * 1024 * 1024
        if size_bytes > max_bytes:
            mb_actual = size_bytes / (1024 * 1024)
            return None, (
                f"File size ({mb_actual:.1f} MB) exceeds the allowable upload limit "
                f"of {max_size_mb:.0f} MB."
            )
        if size_bytes == 0 and not isinstance(file_or_path, str):
            return None, "The uploaded file is empty (0 bytes)."
    except Exception as e:
        return None, f"Unable to verify file size: {str(e)}"

    # 2. Reading CSV with encoding fallbacks
    encodings_to_try = ["utf-8", "utf-8-sig", "latin1", "cp1252"]
    df = None
    last_error = None

    for enc in encodings_to_try:
        try:
            if hasattr(file_or_path, "seek"):
                file_or_path.seek(0)
            df = pd.read_csv(file_or_path, encoding=enc)
            break
        except pd.errors.EmptyDataError:
            return None, "The CSV file contains no readable data or header row."
        except pd.errors.ParserError as pe:
            return None, f"CSV parsing error: {str(pe)}. Please check for malformed delimiters or unescaped quotes."
        except (UnicodeDecodeError, UnicodeError):
            continue
        except Exception as ex:
            last_error = str(ex)
            break

    if df is None:
        return None, (
            f"Could not decode the CSV file. Please ensure it is saved in UTF-8 format. "
            f"Details: {last_error or 'Unknown encoding error'}"
        )

    # 3. Shape validation
    if df.empty:
        if len(df.columns) == 0:
            return None, "The file contains no columns or rows."
        return None, "The CSV file has headers but contains zero data rows."

    return df, None


def infer_column_type(series: pd.Series) -> str:
    """Classifies a Pandas Series into a human-friendly data type category."""
    if series.empty or series.isna().all():
        return "Empty"
    dtype_str = str(series.dtype).lower()
    if "int" in dtype_str or "float" in dtype_str:
        return "Numeric"
    if "datetime" in dtype_str:
        return "Datetime"
    if "bool" in dtype_str:
        return "Boolean"

    # For object/string series, check if it can reasonably parse as datetime or numeric
    non_null = series.dropna().astype(str).str.strip()
    if non_null.empty:
        return "Text"

    # Check date likelihood if strings match ISO or common date formats
    sample = non_null.head(20)
    # Quick date pattern test without full slow parse
    try:
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            parsed_dates = pd.to_datetime(sample, errors="coerce", format="mixed")
        if parsed_dates.notna().sum() / len(sample) >= 0.8:
            return "Date Candidate"
    except Exception:
        pass

    return "Text"


def profile_data(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Computes comprehensive data quality and health metrics for the given dataset.
    Never alters the input DataFrame.
    """
    total_rows = len(df)
    total_cols = len(df.columns)
    duplicate_rows = int(df.duplicated().sum())
    total_cells = total_rows * total_cols
    total_missing = int(df.isna().sum().sum())

    # Count blank/whitespace-only strings in object columns
    blank_strings_total = 0
    column_stats = []

    for col in df.columns:
        s = df[col]
        missing_count = int(s.isna().sum())
        missing_pct = round((missing_count / total_rows) * 100, 1) if total_rows > 0 else 0.0
        unique_count = int(s.nunique(dropna=True))

        # Blank string detection for text/object columns
        blank_count = 0
        if s.dtype == "object" or pd.api.types.is_string_dtype(s):
            # Non-null values that are only whitespace
            non_null_mask = s.notna()
            blank_mask = non_null_mask & (s.astype(str).str.strip() == "")
            blank_count = int(blank_mask.sum())
            blank_strings_total += blank_count

        # Preview sample non-null values
        valid_samples = s.dropna().astype(str).unique()[:3].tolist()
        sample_str = ", ".join(valid_samples) if valid_samples else "None"
        if len(sample_str) > 40:
            sample_str = sample_str[:37] + "..."

        column_stats.append({
            "Column": col,
            "Type": str(s.dtype),
            "Category": infer_column_type(s),
            "Missing": missing_count,
            "Missing %": missing_pct,
            "Blank Strings": blank_count,
            "Unique Values": unique_count,
            "Sample Values": sample_str,
        })

    col_stats_df = pd.DataFrame(column_stats)

    cleanliness_score = round(
        ((total_cells - total_missing - blank_strings_total) / max(total_cells, 1)) * 100, 1
    )

    return {
        "total_rows": total_rows,
        "total_columns": total_cols,
        "total_cells": total_cells,
        "duplicate_rows": duplicate_rows,
        "total_missing": total_missing,
        "missing_pct": round((total_missing / max(total_cells, 1)) * 100, 1),
        "blank_strings_total": blank_strings_total,
        "cleanliness_score": max(0.0, min(100.0, cleanliness_score)),
        "column_stats": col_stats_df,
    }


# =====================================================================
# Cleaning Operations: Preview and Execution
# Every operation operates on a copy and never mutates raw data.
# =====================================================================

def preview_remove_duplicates(df: pd.DataFrame) -> Dict[str, Any]:
    """Previews duplicate rows and provides a preview slice."""
    dup_mask = df.duplicated(keep="first")
    dup_count = int(dup_mask.sum())
    dup_sample = df[dup_mask].head(10)
    return {
        "duplicate_count": dup_count,
        "preview_df": dup_sample,
    }


def apply_remove_duplicates(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Removes exact duplicate rows and returns (cleaned_df, receipt_entry)."""
    initial_count = len(df)
    cleaned_df = df.drop_duplicates(keep="first").reset_index(drop=True)
    affected = initial_count - len(cleaned_df)

    receipt_entry = {
        "timestamp": datetime.now().strftime("%H:%M:%S"),
        "operation": "Remove Duplicate Rows",
        "column": "All Columns",
        "affected_count": affected,
        "explanation": f"Removed {affected} exact duplicate row(s). Row count changed from {initial_count} to {len(cleaned_df)}.",
    }
    return cleaned_df, receipt_entry


def preview_whitespace_trim(
    df: pd.DataFrame,
    columns: Optional[List[str]] = None
) -> Dict[str, Any]:
    """Identifies text columns and counts cells having untrimmed whitespace."""
    target_cols = columns or [c for c in df.columns if df[c].dtype == "object" or pd.api.types.is_string_dtype(df[c])]
    affected_cells = 0
    affected_cols = []
    samples = []

    for col in target_cols:
        s = df[col]
        non_null_mask = s.notna()
        str_series = s[non_null_mask].astype(str)
        needs_trim = str_series != str_series.str.strip()
        count = int(needs_trim.sum())
        if count > 0:
            affected_cells += count
            affected_cols.append(col)
            # Find a few examples
            diff_indices = str_series[needs_trim].index[:3]
            for idx in diff_indices:
                orig_val = str(df.loc[idx, col])
                samples.append({
                    "Column": col,
                    "Row": idx + 1,
                    "Original": f"'{orig_val}'",
                    "Trimmed": f"'{orig_val.strip()}'",
                })

    return {
        "target_columns": target_cols,
        "affected_cells": affected_cells,
        "affected_columns": affected_cols,
        "samples_df": pd.DataFrame(samples) if samples else pd.DataFrame(),
    }


def apply_whitespace_trim(
    df: pd.DataFrame,
    columns: Optional[List[str]] = None
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Trims whitespace from string columns."""
    cleaned_df = df.copy()
    target_cols = columns or [c for c in df.columns if df[c].dtype == "object" or pd.api.types.is_string_dtype(df[c])]
    affected_cells = 0
    cols_modified = []

    for col in target_cols:
        s = cleaned_df[col]
        non_null = s.notna()
        str_series = s[non_null].astype(str)
        needs_trim = str_series != str_series.str.strip()
        count = int(needs_trim.sum())
        if count > 0:
            affected_cells += count
            cols_modified.append(col)
            cleaned_df.loc[non_null, col] = str_series.str.strip()

    receipt_entry = {
        "timestamp": datetime.now().strftime("%H:%M:%S"),
        "operation": "Trim Whitespace",
        "column": ", ".join(cols_modified) if cols_modified else "None",
        "affected_count": affected_cells,
        "explanation": (
            f"Trimmed leading and trailing whitespace from {affected_cells} cell(s) across "
            f"{len(cols_modified)} column(s)."
        ),
    }
    return cleaned_df, receipt_entry


def preview_blank_strings(
    df: pd.DataFrame,
    columns: Optional[List[str]] = None
) -> Dict[str, Any]:
    """Detects blank or whitespace-only strings that should be treated as NaN."""
    target_cols = columns or [c for c in df.columns if df[c].dtype == "object" or pd.api.types.is_string_dtype(df[c])]
    affected_cells = 0
    affected_cols = []
    samples = []

    for col in target_cols:
        s = df[col]
        non_null = s.notna()
        str_series = s[non_null].astype(str)
        blank_mask = str_series.str.strip() == ""
        count = int(blank_mask.sum())
        if count > 0:
            affected_cells += count
            affected_cols.append(col)
            for idx in str_series[blank_mask].index[:3]:
                samples.append({
                    "Column": col,
                    "Row": idx + 1,
                    "Current Value": f"'{str_series.loc[idx]}'",
                    "Proposed Value": "NaN (Missing)",
                })

    return {
        "target_columns": target_cols,
        "affected_cells": affected_cells,
        "affected_columns": affected_cols,
        "samples_df": pd.DataFrame(samples) if samples else pd.DataFrame(),
    }


def apply_blank_strings_to_nan(
    df: pd.DataFrame,
    columns: Optional[List[str]] = None
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Converts blank or whitespace-only text values into formal NaN."""
    cleaned_df = df.copy()
    target_cols = columns or [c for c in df.columns if df[c].dtype == "object" or pd.api.types.is_string_dtype(df[c])]
    affected_cells = 0
    cols_modified = []

    for col in target_cols:
        s = cleaned_df[col]
        non_null = s.notna()
        str_series = s[non_null].astype(str)
        blank_mask = non_null & (cleaned_df[col].astype(str).str.strip() == "")
        count = int(blank_mask.sum())
        if count > 0:
            affected_cells += count
            cols_modified.append(col)
            cleaned_df.loc[blank_mask, col] = np.nan

    receipt_entry = {
        "timestamp": datetime.now().strftime("%H:%M:%S"),
        "operation": "Treat Blank Strings as Missing (NaN)",
        "column": ", ".join(cols_modified) if cols_modified else "None",
        "affected_count": affected_cells,
        "explanation": (
            f"Converted {affected_cells} blank or whitespace-only text cell(s) to formal NaN missing values "
            f"in {len(cols_modified)} column(s)."
        ),
    }
    return cleaned_df, receipt_entry


def preview_casing_normalization(
    df: pd.DataFrame,
    columns: List[str],
    style: str
) -> Dict[str, Any]:
    """Previews capitalization normalization for selected columns."""
    samples = []
    affected_cells = 0

    for col in columns:
        if col not in df.columns:
            continue
        s = df[col]
        non_null = s.notna()
        str_series = s[non_null].astype(str)

        if style == "Title Case":
            transformed = str_series.str.title()
        elif style == "UPPERCASE":
            transformed = str_series.str.upper()
        elif style == "lowercase":
            transformed = str_series.str.lower()
        else:
            transformed = str_series

        diff_mask = str_series != transformed
        count = int(diff_mask.sum())
        affected_cells += count

        for idx in str_series[diff_mask].index[:3]:
            samples.append({
                "Column": col,
                "Row": idx + 1,
                "Original": str_series.loc[idx],
                "Standardized": transformed.loc[idx],
            })

    return {
        "affected_cells": affected_cells,
        "style": style,
        "samples_df": pd.DataFrame(samples) if samples else pd.DataFrame(),
    }


def apply_casing_normalization(
    df: pd.DataFrame,
    columns: List[str],
    style: str
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Applies capitalization normalization (Title Case, UPPERCASE, lowercase)."""
    cleaned_df = df.copy()
    affected_cells = 0
    cols_modified = []

    for col in columns:
        if col not in cleaned_df.columns:
            continue
        s = cleaned_df[col]
        non_null = s.notna()
        str_series = s[non_null].astype(str)

        if style == "Title Case":
            transformed = str_series.str.title()
        elif style == "UPPERCASE":
            transformed = str_series.str.upper()
        elif style == "lowercase":
            transformed = str_series.str.lower()
        else:
            transformed = str_series

        diff_mask = str_series != transformed
        count = int(diff_mask.sum())
        if count > 0:
            affected_cells += count
            cols_modified.append(col)
            cleaned_df.loc[non_null, col] = transformed

    receipt_entry = {
        "timestamp": datetime.now().strftime("%H:%M:%S"),
        "operation": f"Normalize Text Casing ({style})",
        "column": ", ".join(cols_modified) if cols_modified else "None",
        "affected_count": affected_cells,
        "explanation": (
            f"Normalized capitalization to {style} across {affected_cells} cell(s) in "
            f"{len(cols_modified)} column(s)."
        ),
    }
    return cleaned_df, receipt_entry


def preview_missing_handling(
    df: pd.DataFrame,
    columns: List[str],
    strategy: str,
    custom_fill_val: Optional[str] = None
) -> Dict[str, Any]:
    """Previews missing value handling action without mutating data."""
    if not columns:
        return {"affected_count": 0, "affected_rows": 0, "preview_message": "No columns selected."}

    subset_df = df[columns]
    missing_mask = subset_df.isna()
    affected_cells = int(missing_mask.sum().sum())
    rows_affected = int(missing_mask.any(axis=1).sum())

    details = []
    if strategy == "Remove affected rows":
        msg = f"Removing rows with missing values in selected columns will drop {rows_affected} row(s) out of {len(df)}."
    elif strategy == "Fill numeric with column median":
        num_cols = [c for c in columns if pd.api.types.is_numeric_dtype(df[c])]
        if not num_cols:
            msg = "None of the selected columns are numeric. No median fill can be performed."
        else:
            median_info = []
            for c in num_cols:
                m_count = int(df[c].isna().sum())
                if m_count > 0:
                    med = df[c].median()
                    median_info.append(f"'{c}' (median: {med}, {m_count} missing)")
            msg = f"Will fill numeric missing values: {', '.join(median_info) if median_info else 'No missing values in numeric columns'}."
    elif strategy == "Fill text with custom value":
        fill_str = custom_fill_val if custom_fill_val is not None else "Unknown"
        msg = f"Will fill {affected_cells} missing cell(s) across {len(columns)} column(s) with literal value '{fill_str}'."
    else:
        msg = "No changes selected."

    return {
        "affected_cells": affected_cells,
        "affected_rows": rows_affected,
        "message": msg,
    }


def apply_missing_handling(
    df: pd.DataFrame,
    columns: List[str],
    strategy: str,
    custom_fill_val: Optional[str] = None
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Applies chosen missing value handling strategy."""
    cleaned_df = df.copy()

    if strategy == "Remove affected rows":
        initial_len = len(cleaned_df)
        cleaned_df = cleaned_df.dropna(subset=columns).reset_index(drop=True)
        dropped = initial_len - len(cleaned_df)
        receipt_entry = {
            "timestamp": datetime.now().strftime("%H:%M:%S"),
            "operation": "Remove Rows with Missing Values",
            "column": ", ".join(columns),
            "affected_count": dropped,
            "explanation": f"Dropped {dropped} row(s) containing missing values in columns: {', '.join(columns)}.",
        }
        return cleaned_df, receipt_entry

    elif strategy == "Fill numeric with column median":
        num_cols = [c for c in columns if pd.api.types.is_numeric_dtype(cleaned_df[c])]
        filled_total = 0
        details = []
        for c in num_cols:
            missing_count = int(cleaned_df[c].isna().sum())
            if missing_count > 0:
                med = cleaned_df[c].median()
                cleaned_df[c] = cleaned_df[c].fillna(med)
                filled_total += missing_count
                details.append(f"{c} (median={med:.2f}, {missing_count} cells)")

        explanation = (
            f"Filled {filled_total} missing numeric cell(s) using column medians: {', '.join(details)}."
            if details else "No missing numeric values found to fill."
        )
        receipt_entry = {
            "timestamp": datetime.now().strftime("%H:%M:%S"),
            "operation": "Impute Numeric Median",
            "column": ", ".join(num_cols) if num_cols else "None",
            "affected_count": filled_total,
            "explanation": explanation,
        }
        return cleaned_df, receipt_entry

    elif strategy == "Fill text with custom value":
        fill_val = custom_fill_val if (custom_fill_val is not None and custom_fill_val != "") else "Unknown"
        filled_total = 0
        for c in columns:
            missing_count = int(cleaned_df[c].isna().sum())
            if missing_count > 0:
                cleaned_df[c] = cleaned_df[c].fillna(fill_val)
                filled_total += missing_count

        receipt_entry = {
            "timestamp": datetime.now().strftime("%H:%M:%S"),
            "operation": "Fill Text Missing Values",
            "column": ", ".join(columns),
            "affected_count": filled_total,
            "explanation": f"Replaced {filled_total} missing value(s) in {len(columns)} column(s) with '{fill_val}'.",
        }
        return cleaned_df, receipt_entry

    else:
        # Leave unchanged
        receipt_entry = {
            "timestamp": datetime.now().strftime("%H:%M:%S"),
            "operation": "Missing Values Unchanged",
            "column": ", ".join(columns),
            "affected_count": 0,
            "explanation": "No missing value modifications were applied per user selection.",
        }
        return cleaned_df, receipt_entry


def preview_type_conversion(
    df: pd.DataFrame,
    column: str,
    target_type: str
) -> Dict[str, Any]:
    """
    Evaluates whether conversion of a column to Numeric or Date is feasible.
    Reports valid conversions vs unparseable values so the user is never surprised.
    """
    s = df[column]
    non_null = s.dropna()
    total_non_null = len(non_null)
    if total_non_null == 0:
        return {
            "column": column,
            "target_type": target_type,
            "total_non_null": 0,
            "valid_count": 0,
            "invalid_count": 0,
            "invalid_sample": [],
            "feasible": False,
            "message": "Column has no non-null values to convert."
        }

    if target_type == "Numeric":
        # Test pd.to_numeric
        converted = pd.to_numeric(non_null.astype(str).str.strip(), errors="coerce")
        valid_count = int(converted.notna().sum())
        invalid_count = total_non_null - valid_count
        invalid_sample = non_null[converted.isna()].head(5).tolist()
    elif target_type == "Datetime":
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            converted = pd.to_datetime(non_null.astype(str).str.strip(), errors="coerce", format="mixed")
        valid_count = int(converted.notna().sum())
        invalid_count = total_non_null - valid_count
        invalid_sample = non_null[converted.isna()].head(5).tolist()
    else:
        return {"feasible": False, "message": f"Unsupported target type: {target_type}"}

    pct_valid = (valid_count / total_non_null) * 100
    feasible = pct_valid >= 50.0  # reasonable threshold

    return {
        "column": column,
        "target_type": target_type,
        "total_non_null": total_non_null,
        "valid_count": valid_count,
        "invalid_count": invalid_count,
        "invalid_sample": invalid_sample,
        "pct_valid": round(pct_valid, 1),
        "feasible": feasible,
        "message": (
            f"{valid_count} of {total_non_null} ({pct_valid:.1f}%) values can be converted successfully. "
            f"{invalid_count} unparseable value(s) will be set to NaN."
            if invalid_count > 0 else
            f"All {valid_count} values can be cleanly converted to {target_type}."
        )
    }


def apply_type_conversion(
    df: pd.DataFrame,
    column: str,
    target_type: str
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Converts a column to Numeric or Datetime, reporting any invalid coercions."""
    cleaned_df = df.copy()
    s = cleaned_df[column]
    non_null_before = int(s.notna().sum())

    if target_type == "Numeric":
        converted = pd.to_numeric(s.astype(str).str.strip(), errors="coerce")
        cleaned_df[column] = converted
        valid_count = int(converted.notna().sum())
        invalid_count = max(0, non_null_before - valid_count)
    elif target_type == "Datetime":
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            converted = pd.to_datetime(s.astype(str).str.strip(), errors="coerce", format="mixed")
        cleaned_df[column] = converted
        valid_count = int(converted.notna().sum())
        invalid_count = max(0, non_null_before - valid_count)
    else:
        raise ValueError(f"Unsupported target type {target_type}")

    receipt_entry = {
        "timestamp": datetime.now().strftime("%H:%M:%S"),
        "operation": f"Convert Type to {target_type}",
        "column": column,
        "affected_count": valid_count,
        "explanation": (
            f"Converted column '{column}' to {target_type}. {valid_count} value(s) successfully converted. "
            f"{invalid_count} unparseable value(s) converted to NaN."
        ),
    }
    return cleaned_df, receipt_entry


# =====================================================================
# Cleaning Receipt Management
# =====================================================================

class CleaningReceiptTracker:
    """
    Maintains an audit trail of every data cleaning step performed.
    Computes before-and-after differences in row counts, column counts, and missing values.
    """

    def __init__(self, raw_df: pd.DataFrame):
        self.raw_rows = len(raw_df)
        self.raw_cols = len(raw_df.columns)
        self.raw_missing = int(raw_df.isna().sum().sum())
        self.raw_duplicates = int(raw_df.duplicated().sum())
        self.operations: List[Dict[str, Any]] = []

    def record(self, entry: Dict[str, Any]):
        """Records an operation receipt entry."""
        entry_copy = dict(entry)
        entry_copy["step"] = len(self.operations) + 1
        if "timestamp" not in entry_copy:
            entry_copy["timestamp"] = datetime.now().strftime("%H:%M:%S")
        self.operations.append(entry_copy)

    def get_summary(self, current_df: pd.DataFrame) -> Dict[str, Any]:
        """Calculates before vs. after statistics."""
        current_rows = len(current_df)
        current_cols = len(current_df.columns)
        current_missing = int(current_df.isna().sum().sum())
        current_duplicates = int(current_df.duplicated().sum())

        return {
            "initial_rows": self.raw_rows,
            "cleaned_rows": current_rows,
            "rows_diff": current_rows - self.raw_rows,
            "initial_cols": self.raw_cols,
            "cleaned_cols": current_cols,
            "initial_missing": self.raw_missing,
            "cleaned_missing": current_missing,
            "missing_diff": current_missing - self.raw_missing,
            "initial_duplicates": self.raw_duplicates,
            "cleaned_duplicates": current_duplicates,
            "operations_count": len(self.operations),
        }

    def generate_text_receipt(self, current_df: pd.DataFrame) -> str:
        """Generates a clean, human-readable text audit report."""
        summary = self.get_summary(current_df)
        lines = [
            "============================================================",
            "                   DATALENS CLEANING RECEIPT                 ",
            "         Clean your data. Understand your numbers.          ",
            "============================================================",
            f"Generated At       : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"Operations Applied : {summary['operations_count']}",
            "------------------------------------------------------------",
            "DATASET SUMMARY COMPARISON",
            "------------------------------------------------------------",
            f"Rows               : {summary['initial_rows']} -> {summary['cleaned_rows']} (Diff: {summary['rows_diff']:+d})",
            f"Columns            : {summary['initial_cols']} -> {summary['cleaned_cols']}",
            f"Missing Values     : {summary['initial_missing']} -> {summary['cleaned_missing']} (Diff: {summary['missing_diff']:+d})",
            f"Duplicate Rows     : {summary['initial_duplicates']} -> {summary['cleaned_duplicates']}",
            "------------------------------------------------------------",
            "CHRONOLOGICAL LOG OF OPERATIONS",
            "------------------------------------------------------------",
        ]

        if not self.operations:
            lines.append("No cleaning operations have been applied yet.")
        else:
            for op in self.operations:
                lines.append(f"Step {op.get('step', 1)} [{op.get('timestamp', '--:--:--')}] {op.get('operation', 'Operation')}")
                lines.append(f"  Target Column(s) : {op.get('column', 'All')}")
                lines.append(f"  Affected Count   : {op.get('affected_count', 0)}")
                lines.append(f"  Explanation      : {op.get('explanation', '')}")
                lines.append("")

        lines.extend([
            "============================================================",
            "End of Cleaning Receipt. Preserved by DataLens.",
            "============================================================",
        ])
        return "\n".join(lines)

    def generate_csv_receipt(self) -> str:
        """Exports the operations log as a structured CSV string."""
        if not self.operations:
            df_receipt = pd.DataFrame(columns=["step", "timestamp", "operation", "column", "affected_count", "explanation"])
        else:
            df_receipt = pd.DataFrame(self.operations)
        return df_receipt.to_csv(index=False)


# =====================================================================
# Factual Insights Engine
# Generates transparent, verifiable observations using pure statistics.
# =====================================================================

def generate_insights(df: pd.DataFrame, raw_df: Optional[pd.DataFrame] = None) -> List[Dict[str, Any]]:
    """
    Computes factual, transparent data observations supported directly by the numbers.
    Never claims to use AI; provides statistical flags with clear explanations.
    """
    insights = []
    total_rows = len(df)
    if total_rows == 0:
        return [{"title": "Empty Dataset", "detail": "No rows available to evaluate.", "type": "info"}]

    # 1. Missing Data Observation
    missing_by_col = df.isna().sum()
    total_missing = int(missing_by_col.sum())
    if total_missing > 0:
        worst_col = missing_by_col.idxmax()
        worst_count = int(missing_by_col[worst_col])
        worst_pct = (worst_count / total_rows) * 100
        insights.append({
            "category": "Data Completeness",
            "badge": "Missing Values",
            "title": f"Column '{worst_col}' has the highest missingness",
            "detail": (
                f"Contains {worst_count} missing value(s) ({worst_pct:.1f}% of all rows in this column). "
                f"Overall, the dataset has {total_missing} missing value(s) across all fields."
            ),
            "type": "warning" if worst_pct > 15 else "info",
        })
    else:
        insights.append({
            "category": "Data Completeness",
            "badge": "100% Complete",
            "title": "Zero missing values detected",
            "detail": f"Every cell across all {total_rows} row(s) and {len(df.columns)} column(s) contains valid data.",
            "type": "success",
        })

    # 2. Categorical Imbalance / Mode
    cat_cols = [c for c in df.columns if df[c].dtype == "object" or pd.api.types.is_string_dtype(df[c])]
    if cat_cols:
        best_cat_col = None
        max_mode_pct = -1.0
        mode_val = None
        mode_freq = 0

        for c in cat_cols:
            s_clean = df[c].dropna()
            if not s_clean.empty:
                val_counts = s_clean.value_counts()
                top_freq = val_counts.iloc[0]
                pct = (top_freq / len(s_clean)) * 100
                if pct > max_mode_pct and val_counts.shape[0] > 1:
                    max_mode_pct = pct
                    best_cat_col = c
                    mode_val = val_counts.index[0]
                    mode_freq = top_freq

        if best_cat_col and max_mode_pct >= 25.0:
            insights.append({
                "category": "Categorical Distribution",
                "badge": "Dominant Category",
                "title": f"'{mode_val}' represents {max_mode_pct:.1f}% of '{best_cat_col}'",
                "detail": (
                    f"In column '{best_cat_col}', the most frequent entry is '{mode_val}', "
                    f"appearing {mode_freq} time(s) out of {len(df[best_cat_col].dropna())} valid entries."
                ),
                "type": "info",
            })

    # 3. Numeric Dispersion & Statistical Outlier Flag (IQR Rule)
    num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    outlier_found = False

    for c in num_cols:
        s = df[c].dropna()
        if len(s) >= 4:
            q1 = float(s.quantile(0.25))
            q3 = float(s.quantile(0.75))
            iqr = q3 - q1
            upper_fence = q3 + 1.5 * iqr
            lower_fence = q1 - 1.5 * iqr

            outliers_high = s[s > upper_fence]
            outliers_low = s[s < lower_fence]
            total_outliers = len(outliers_high) + len(outliers_low)

            if total_outliers > 0 and not outlier_found:
                outlier_found = True
                max_val = float(s.max())
                median_val = float(s.median())
                mean_val = float(s.mean())

                detail_msg = (
                    f"Statistical Flag: Detected {total_outliers} value(s) in '{c}' outside the standard 1.5×IQR boundaries "
                    f"(interquartile range: {q1:.2f} to {q3:.2f}). Upper threshold is {upper_fence:.2f}; "
                    f"maximum value is {max_val:.2f} (median: {median_val:.2f}, mean: {mean_val:.2f}). "
                    f"*Note: This is an objective statistical flag based on dispersion, not proof that the record is incorrect.*"
                )
                insights.append({
                    "category": "Statistical Dispersion",
                    "badge": "Statistical Flag",
                    "title": f"Unusually high or low values detected in '{c}'",
                    "detail": detail_msg,
                    "type": "warning",
                })

    # 4. Primary Key / Unique Identifier Check
    id_candidates = []
    for c in df.columns:
        s = df[c]
        if s.isna().sum() == 0 and s.nunique() == total_rows and total_rows > 1:
            id_candidates.append(c)

    if id_candidates:
        insights.append({
            "category": "Schema Integrity",
            "badge": "Unique Identifier",
            "title": f"Column '{id_candidates[0]}' has 100% unique values",
            "detail": (
                f"Every row has a distinct, non-missing entry in '{id_candidates[0]}'. "
                f"This field is well-suited to serve as a primary key or unique identifier."
            ),
            "type": "success",
        })

    # 5. Cleanliness comparison if raw_df is provided
    if raw_df is not None:
        raw_rows = len(raw_df)
        raw_missing = int(raw_df.isna().sum().sum())
        rows_saved = raw_rows - total_rows
        missing_reduced = raw_missing - total_missing

        if rows_saved > 0 or missing_reduced > 0:
            insights.append({
                "category": "Cleaning Progress",
                "badge": "Data Quality Boost",
                "title": f"Cleaning improved dataset health",
                "detail": (
                    f"Operations reduced missing values by {missing_reduced} "
                    f"and resolved {rows_saved} unnecessary/duplicate row(s)."
                ),
                "type": "success",
            })

    return insights
