"""
verify_end_to_end.py - Comprehensive End-to-End Workflow Verification for DataLens
Tests all user stories, edge cases, receipts, and insights against real datasets.
"""

import io
import os
import sys
import pandas as pd
import numpy as np

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

def run_all_verifications():
    print("=" * 60)
    print("RUNNING DATALENS END-TO-END VERIFICATION SUITE")
    print("=" * 60)

    # -------------------------------------------------------------
    # 1. Verify Sample Business Dataset Loading & Profiling
    # -------------------------------------------------------------
    print("\n[Step 1] Loading sample_data.csv...")
    assert os.path.exists("sample_data.csv"), "sample_data.csv does not exist"
    df_raw, err = load_data("sample_data.csv")
    assert err is None, f"Failed to load sample_data.csv: {err}"
    assert df_raw is not None
    initial_rows = len(df_raw)
    initial_cols = len(df_raw.columns)
    print(f"✓ Loaded sample_data.csv successfully: {initial_rows} rows, {initial_cols} columns")

    profile_init = profile_data(df_raw)
    print(f"  - Duplicate rows detected: {profile_init['duplicate_rows']}")
    print(f"  - Missing cells detected: {profile_init['total_missing']}")
    print(f"  - Blank text strings detected: {profile_init['blank_strings_total']}")
    print(f"  - Baseline cleanliness score: {profile_init['cleanliness_score']}%")
    assert profile_init['duplicate_rows'] == 3, f"Expected 3 duplicate rows, got {profile_init['duplicate_rows']}"
    assert profile_init['blank_strings_total'] >= 2, "Expected blank strings detected"

    # Initialize Receipt Tracker
    tracker = CleaningReceiptTracker(df_raw)
    curr_df = df_raw.copy()

    # -------------------------------------------------------------
    # 2. Verify Duplicate Removal
    # -------------------------------------------------------------
    print("\n[Step 2] Testing duplicate removal...")
    dup_preview = preview_remove_duplicates(curr_df)
    assert dup_preview["duplicate_count"] == 3
    curr_df, r1 = apply_remove_duplicates(curr_df)
    tracker.record(r1)
    assert len(curr_df) == initial_rows - 3
    assert r1["affected_count"] == 3
    print(f"✓ Removed 3 duplicates. Rows now: {len(curr_df)}")

    # -------------------------------------------------------------
    # 3. Verify Whitespace Trimming
    # -------------------------------------------------------------
    print("\n[Step 3] Testing whitespace trimming...")
    ws_preview = preview_whitespace_trim(curr_df)
    print(f"  - Cells needing trimming: {ws_preview['affected_cells']}")
    assert ws_preview["affected_cells"] > 0
    curr_df, r2 = apply_whitespace_trim(curr_df)
    tracker.record(r2)
    print(f"✓ Trimmed whitespace from {r2['affected_count']} cells across: {r2['column']}")

    # -------------------------------------------------------------
    # 4. Verify Blank Strings to NaN
    # -------------------------------------------------------------
    print("\n[Step 4] Testing conversion of blank strings to NaN...")
    bs_preview = preview_blank_strings(curr_df)
    print(f"  - Blank cells found: {bs_preview['affected_cells']}")
    assert bs_preview["affected_cells"] >= 2
    curr_df, r3 = apply_blank_strings_to_nan(curr_df)
    tracker.record(r3)
    print(f"✓ Converted {r3['affected_count']} blank cells to NaN.")

    # -------------------------------------------------------------
    # 5. Verify Capitalization Normalization
    # -------------------------------------------------------------
    print("\n[Step 5] Testing casing normalization on 'industry'...")
    case_preview = preview_casing_normalization(curr_df, ["industry"], "Title Case")
    assert case_preview["affected_cells"] > 0
    curr_df, r4 = apply_casing_normalization(curr_df, ["industry"], "Title Case")
    tracker.record(r4)
    # Verify no lowercase 'technology' or 'TECHNOLOGY' remains
    unique_industries = curr_df["industry"].dropna().unique().tolist()
    assert "technology" not in unique_industries
    assert "TECHNOLOGY" not in unique_industries
    assert "Technology" in unique_industries
    print(f"✓ Standardized 'industry' to Title Case. Affected: {r4['affected_count']} cells.")

    # -------------------------------------------------------------
    # 6. Verify Type Conversion (order_date to Datetime)
    # -------------------------------------------------------------
    print("\n[Step 6] Testing date conversion on 'order_date'...")
    date_preview = preview_type_conversion(curr_df, "order_date", "Datetime")
    assert date_preview["feasible"] is True
    curr_df, r5 = apply_type_conversion(curr_df, "order_date", "Datetime")
    tracker.record(r5)
    assert pd.api.types.is_datetime64_any_dtype(curr_df["order_date"])
    print(f"✓ Converted 'order_date' to Datetime. Converted count: {r5['affected_count']}")

    # -------------------------------------------------------------
    # 7. Verify Missing Value Imputation
    # -------------------------------------------------------------
    print("\n[Step 7] Testing numeric median imputation...")
    curr_df, r6 = apply_missing_handling(curr_df, ["quantity"], "Fill numeric with column median")
    tracker.record(r6)
    assert curr_df["quantity"].isna().sum() == 0
    print(f"✓ Filled missing quantities with median: {r6['affected_count']} cells filled.")

    print("\n[Step 7b] Testing text fill with custom value for 'payment_status'...")
    curr_df, r7 = apply_missing_handling(curr_df, ["payment_status"], "Fill text with custom value", "Unspecified")
    tracker.record(r7)
    assert curr_df["payment_status"].isna().sum() == 0
    assert "Unspecified" in curr_df["payment_status"].values
    print(f"✓ Replaced missing payment status with 'Unspecified': {r7['affected_count']} cells filled.")

    # -------------------------------------------------------------
    # 8. Verify Cleaning Receipt Audit Trail
    # -------------------------------------------------------------
    print("\n[Step 8] Verifying Cleaning Receipt & Diffs...")
    summary = tracker.get_summary(curr_df)
    assert summary["initial_rows"] == initial_rows
    assert summary["cleaned_rows"] == initial_rows - 3
    assert summary["rows_diff"] == -3
    assert summary["operations_count"] == 7
    print(f"✓ Summary comparison: Initial {summary['initial_rows']} rows -> Cleaned {summary['cleaned_rows']} rows.")

    text_receipt = tracker.generate_text_receipt(curr_df)
    assert "DATALENS CLEANING RECEIPT" in text_receipt
    assert "Remove Duplicate Rows" in text_receipt
    assert "Trim Whitespace" in text_receipt
    assert "Normalize Text Casing" in text_receipt
    assert "Impute Numeric Median" in text_receipt
    print("✓ Full text audit receipt generated successfully.")

    csv_receipt = tracker.generate_csv_receipt()
    assert "operation,column,affected_count" in csv_receipt
    print("✓ CSV audit log generated successfully.")

    # -------------------------------------------------------------
    # 9. Verify Factual Insights Generation
    # -------------------------------------------------------------
    print("\n[Step 9] Verifying Factual Insights...")
    insights = generate_insights(curr_df, df_raw)
    assert len(insights) >= 3
    for ins in insights:
        print(f"  - [{ins.get('badge')}] {ins.get('title')}")
        assert "AI" not in ins["detail"], "AI hallucination claim found in insights"
    print("✓ Factual statistical insights verified.")

    # -------------------------------------------------------------
    # 10. Verify Raw Data Immutability & Reset Capability
    # -------------------------------------------------------------
    print("\n[Step 10] Verifying raw data preservation...")
    assert len(df_raw) == initial_rows, "Raw DataFrame was mutated during cleaning operations!"
    assert df_raw.duplicated().sum() == 3, "Raw DataFrame lost its duplicate rows!"
    print("✓ Verified raw data is 100% immutable and intact in memory.")

    # -------------------------------------------------------------
    # 11. Verify Secondary Dataset (test_inventory.csv)
    # -------------------------------------------------------------
    print("\n[Step 11] Verifying secondary dataset test_inventory.csv...")
    inv_df, inv_err = load_data("test_inventory.csv")
    assert inv_err is None
    inv_profile = profile_data(inv_df)
    assert inv_profile["total_rows"] == 11
    assert inv_profile["duplicate_rows"] == 1
    inv_cleaned, _ = apply_remove_duplicates(inv_df)
    assert len(inv_cleaned) == 10
    print("✓ Secondary dataset (warehouse inventory) loaded and processed perfectly.")

    # -------------------------------------------------------------
    # 12. Verify Edge Cases (Empty file, headers only, 1 column)
    # -------------------------------------------------------------
    print("\n[Step 12] Testing edge cases (empty files, header-only)...")
    empty_res, empty_err = load_data(io.StringIO(""))
    assert empty_res is None
    assert "empty" in empty_err.lower()

    header_only_res, header_err = load_data(io.StringIO("colA,colB,colC\n"))
    assert header_only_res is None
    assert "zero data rows" in header_err.lower()

    single_col_df, single_err = load_data(io.StringIO("number\n10\n20\n30\n"))
    assert single_err is None
    assert len(single_col_df) == 3
    print("✓ All edge cases handled gracefully with user-friendly error messages.")

    print("\n" + "=" * 60)
    print("ALL VERIFICATIONS COMPLETED SUCCESSFULLY (12/12 PASSED)!")
    print("=" * 60)

if __name__ == "__main__":
    if sys.platform == "win32":
        try:
            sys.stdout.reconfigure(encoding='utf-8')
        except Exception:
            pass
    run_all_verifications()
