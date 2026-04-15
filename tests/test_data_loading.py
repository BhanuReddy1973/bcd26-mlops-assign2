"""
Unit tests for data loading module
"""
import pytest
import pandas as pd
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.load_data import load_data

def test_load_data_returns_dataframe():
    """Test that load_data returns a pandas DataFrame"""
    # This test assumes data exists - in real scenario, use mock data
    try:
        df = load_data("data/raw/customer_churn.csv")
        assert isinstance(df, pd.DataFrame)
    except FileNotFoundError:
        pytest.skip("Test data not available")

def test_load_data_has_expected_columns():
    """Test that loaded data has expected structure"""
    try:
        df = load_data("data/raw/customer_churn.csv")
        expected_cols = ["customerID", "Churn"]
        for col in expected_cols:
            assert col in df.columns
    except FileNotFoundError:
        pytest.skip("Test data not available")
