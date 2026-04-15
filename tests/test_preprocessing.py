"""
Unit tests for preprocessing module
"""
import pytest
import pandas as pd
import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.preprocess import preprocess_data

def test_preprocess_removes_customerid():
    """Test that customerID is removed"""
    df = pd.DataFrame({
        "customerID": ["001", "002"],
        "TotalCharges": ["100", "200"],
        "Churn": ["Yes", "No"]
    })
    result = preprocess_data(df)
    assert "customerID" not in result.columns

def test_preprocess_converts_churn_to_binary():
    """Test that Churn is converted to 0/1"""
    df = pd.DataFrame({
        "customerID": ["001", "002"],
        "TotalCharges": ["100", "200"],
        "Churn": ["Yes", "No"]
    })
    result = preprocess_data(df)
    assert result["Churn"].dtype in [np.int64, np.int32]
    assert set(result["Churn"].unique()).issubset({0, 1})
