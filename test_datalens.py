"""
test_datalens.py - Unit and Integration Tests for DataLens Data Processing Engine
Covers loading, profiling, cleaning, receipts, and statistical insights.
"""

import io
import unittest
import numpy as np
import pandas as pd

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


class TestDataLensEngine(unittest.TestCase):

    def setUp(self):
        """Create a standard test dataset with known issues."""
        self.sample_csv_text = (
            "id,name,category,amount,date\n"
            "1,  Alpha Corp  ,Software,100.50,2024-01-01\n"
            "2,Beta LLC,hardware,250.00,2024-01-02\n"
            "3,Gamma Inc,SOFTWARE, ,2024-01-03\n"
            "4,Delta Co,   ,450.00,2024-01-04\n"
            "5,Epsilon Ltd,Services,12000.00,2024-01-05\n"
            "1,  Alpha Corp  ,Software,100.50,2024-01-01\n"  # Exact duplicate of row 1
        )
        self.sample_buf = io.StringIO(self.sample_csv_text)

    # -----------------------------------------------------------------
    # 1. Loading & Profiling Tests
    # -----------------------------------------------------------------

    def test_load_data_valid(self):
        df, err = load_data(io.StringIO(self.sample_csv_text))
        self.assertIsNone(err)
        self.assertIsNotNone(df)
        self.assertEqual(len(df), 6)
        self.assertEqual(len(df.columns), 5)

    def test_load_data_empty_file(self):
        empty_buf = io.StringIO("")
        df, err = load_data(empty_buf)
        self.assertIsNone(df)
        self.assertIn("empty", err.lower())

    def test_load_data_malformed_csv(self):
        bad_csv = "col1,col2\n1,2,3,4\n5,6\n"
        df, err = load_data(io.StringIO(bad_csv))
        # Pandas or parser should catch or report
        # If pandas fails with ParserError, load_data returns an error message
        if err:
            self.assertIn("error", err.lower())

    def test_profile_data_metrics(self):
        df, _ = load_data(io.StringIO(self.sample_csv_text))
        profile = profile_data(df)

        self.assertEqual(profile["total_rows"], 6)
        self.assertEqual(profile["total_columns"], 5)
        self.assertEqual(profile["duplicate_rows"], 1)
        # Check blank strings detected in 'category' ("   ") and 'amount' (" ")
        self.assertGreater(profile["blank_strings_total"], 0)
        self.assertIn("column_stats", profile)
        self.assertEqual(len(profile["column_stats"]), 5)

    # -----------------------------------------------------------------
    # 2. Cleaning: Duplicate Removal
    # -----------------------------------------------------------------

    def test_duplicates_preview_and_apply(self):
        df, _ = load_data(io.StringIO(self.sample_csv_text))
        preview = preview_remove_duplicates(df)
        self.assertEqual(preview["duplicate_count"], 1)

        cleaned_df, receipt = apply_remove_duplicates(df)
        self.assertEqual(len(cleaned_df), 5)
        self.assertEqual(receipt["affected_count"], 1)
        self.assertIn("Remove Duplicate Rows", receipt["operation"])
        # Original df must be untouched
        self.assertEqual(len(df), 6)

    # -----------------------------------------------------------------
    # 3. Cleaning: Whitespace Trimming
    # -----------------------------------------------------------------

    def test_whitespace_trimming(self):
        df, _ = load_data(io.StringIO(self.sample_csv_text))
        preview = preview_whitespace_trim(df)
        self.assertGreater(preview["affected_cells"], 0)
        self.assertIn("name", preview["affected_columns"])

        cleaned_df, receipt = apply_whitespace_trim(df)
        # Check that leading/trailing space in "  Alpha Corp  " was trimmed
        self.assertEqual(cleaned_df.loc[0, "name"], "Alpha Corp")
        self.assertGreater(receipt["affected_count"], 0)

    # -----------------------------------------------------------------
    # 4. Cleaning: Blank Strings to NaN
    # -----------------------------------------------------------------

    def test_blank_strings_to_nan(self):
        df, _ = load_data(io.StringIO(self.sample_csv_text))
        # Row 3 (Delta Co) has category "   "
        preview = preview_blank_strings(df)
        self.assertGreater(preview["affected_cells"], 0)

        cleaned_df, receipt = apply_blank_strings_to_nan(df)
        self.assertTrue(pd.isna(cleaned_df.loc[3, "category"]))
        self.assertGreater(receipt["affected_count"], 0)

    # -----------------------------------------------------------------
    # 5. Cleaning: Capitalization Normalization
    # -----------------------------------------------------------------

    def test_casing_normalization(self):
        df, _ = load_data(io.StringIO(self.sample_csv_text))
        # 'category' has "hardware", "SOFTWARE", "Services"
        preview = preview_casing_normalization(df, ["category"], "Title Case")
        self.assertGreater(preview["affected_cells"], 0)

        cleaned_df, receipt = apply_casing_normalization(df, ["category"], "Title Case")
        self.assertEqual(cleaned_df.loc[1, "category"], "Hardware")
        self.assertEqual(cleaned_df.loc[2, "category"], "Software")

    # -----------------------------------------------------------------
    # 6. Cleaning: Missing Value Handling
    # -----------------------------------------------------------------

    def test_missing_value_median_fill(self):
        test_df = pd.DataFrame({
            "val": [10.0, 20.0, np.nan, 40.0],
            "text": ["A", "B", np.nan, "D"]
        })
        # Median of [10, 20, 40] is 20.0
        cleaned_df, receipt = apply_missing_handling(test_df, ["val"], "Fill numeric with column median")
        self.assertEqual(cleaned_df["val"].isna().sum(), 0)
        self.assertEqual(cleaned_df.loc[2, "val"], 20.0)
        self.assertEqual(receipt["affected_count"], 1)

    def test_missing_value_custom_text_fill(self):
        test_df = pd.DataFrame({
            "status": ["Active", np.nan, "Pending"]
        })
        cleaned_df, receipt = apply_missing_handling(test_df, ["status"], "Fill text with custom value", "Unassigned")
        self.assertEqual(cleaned_df.loc[1, "status"], "Unassigned")
        self.assertEqual(cleaned_df["status"].isna().sum(), 0)

    def test_missing_value_drop_rows(self):
        test_df = pd.DataFrame({
            "status": ["Active", np.nan, "Pending"]
        })
        cleaned_df, receipt = apply_missing_handling(test_df, ["status"], "Remove affected rows")
        self.assertEqual(len(cleaned_df), 2)
        self.assertEqual(receipt["affected_count"], 1)

    # -----------------------------------------------------------------
    # 7. Cleaning: Type Conversion
    # -----------------------------------------------------------------

    def test_type_conversion_numeric(self):
        test_df = pd.DataFrame({
            "price_str": ["12.50", "99.00", "invalid_val", "45.10"]
        })
        preview = preview_type_conversion(test_df, "price_str", "Numeric")
        self.assertEqual(preview["valid_count"], 3)
        self.assertEqual(preview["invalid_count"], 1)
        self.assertTrue(preview["feasible"])

        cleaned_df, receipt = apply_type_conversion(test_df, "price_str", "Numeric")
        self.assertTrue(pd.api.types.is_numeric_dtype(cleaned_df["price_str"]))
        self.assertTrue(pd.isna(cleaned_df.loc[2, "price_str"]))
        self.assertIn("converted to NaN", receipt["explanation"])

    def test_type_conversion_datetime(self):
        test_df = pd.DataFrame({
            "date_col": ["2024-01-01", "2024-02-15", "not_a_date"]
        })
        preview = preview_type_conversion(test_df, "date_col", "Datetime")
        self.assertEqual(preview["valid_count"], 2)
        self.assertEqual(preview["invalid_count"], 1)

        cleaned_df, receipt = apply_type_conversion(test_df, "date_col", "Datetime")
        self.assertTrue(pd.api.types.is_datetime64_any_dtype(cleaned_df["date_col"]))

    # -----------------------------------------------------------------
    # 8. Cleaning Receipt Tracker
    # -----------------------------------------------------------------

    def test_receipt_tracker(self):
        raw_df, _ = load_data(io.StringIO(self.sample_csv_text))
        tracker = CleaningReceiptTracker(raw_df)

        # Apply duplicate removal
        cleaned_df, r1 = apply_remove_duplicates(raw_df)
        tracker.record(r1)

        summary = tracker.get_summary(cleaned_df)
        self.assertEqual(summary["initial_rows"], 6)
        self.assertEqual(summary["cleaned_rows"], 5)
        self.assertEqual(summary["rows_diff"], -1)
        self.assertEqual(summary["operations_count"], 1)

        text_rep = tracker.generate_text_receipt(cleaned_df)
        self.assertIn("DATALENS CLEANING RECEIPT", text_rep)
        self.assertIn("Remove Duplicate Rows", text_rep)

        csv_rep = tracker.generate_csv_receipt()
        self.assertIn("operation,column,affected_count", csv_rep)

    # -----------------------------------------------------------------
    # 9. Statistical Insights Generation
    # -----------------------------------------------------------------

    def test_generate_insights(self):
        # Create dataset with clear outlier (12000 vs 100, 200, 300)
        df = pd.DataFrame({
            "uid": [1, 2, 3, 4, 5],
            "dept": ["Engineering", "Engineering", "Engineering", "Marketing", "Sales"],
            "salary": [50000, 52000, 51000, 49000, 450000],  # 450000 is statistical outlier
            "missing_col": [1, np.nan, np.nan, 4, 5]
        })
        insights = generate_insights(df)
        self.assertGreater(len(insights), 0)

        # Verify missing column detection
        missing_insight = next((i for i in insights if "missing" in i["title"].lower()), None)
        self.assertIsNotNone(missing_insight)
        self.assertIn("missing_col", missing_insight["title"])

        # Verify outlier detection
        outlier_insight = next((i for i in insights if "statistical flag" in i["badge"].lower()), None)
        self.assertIsNotNone(outlier_insight)
        self.assertIn("salary", outlier_insight["title"])

        # Verify unique ID detection
        id_insight = next((i for i in insights if "unique identifier" in i["badge"].lower()), None)
        self.assertIsNotNone(id_insight)
        self.assertIn("uid", id_insight["title"])

    # -----------------------------------------------------------------
    # 10. Sample File Validation
    # -----------------------------------------------------------------

    def test_sample_business_data_file(self):
        df, err = load_data("sample_data.csv")
        self.assertIsNone(err)
        self.assertIsNotNone(df)
        self.assertGreater(len(df), 20)

        profile = profile_data(df)
        # Should detect duplicates
        self.assertGreaterEqual(profile["duplicate_rows"], 1)
        # Should detect blank strings
        self.assertGreaterEqual(profile["blank_strings_total"], 1)


if __name__ == "__main__":
    unittest.main()
