"""
Unit tests for data_preprocessing.py
"""

import pytest
import pandas as pd
from data_preprocessing import (
    normalize_text,
    remove_stopwords,
    flag_antipatterns,
    preprocess_data
)

@pytest.fixture
def sample_df() -> pd.DataFrame:
    """Provides a sample DataFrame for testing."""
    data = {
        'name': [
            'Sales Report Q1', 
            'Copy of Financials', 
            'test_dashboard',
            'Final Report v2',
            'Customer Analysis'
        ],
        'description': [
            'A report showing Q1 Sales data.', 
            'This is a copy of the main financial report.', 
            '',
            None,
            'Analysis of customer behavior and segmentation.'
        ]
    }
    return pd.DataFrame(data)

def test_normalize_text():
    """Test the normalize_text function."""
    text = "This is a Test!"
    expected = ['this', 'is', 'a', 'test']
    assert normalize_text(text) == expected
    
    text_with_punct = "Another... test, with punctuation;"
    expected_with_punct = ['another', 'test', 'with', 'punctuation']
    assert normalize_text(text_with_punct) == expected_with_punct
    
    assert normalize_text("") == []
    assert normalize_text(None) == []
    assert normalize_text("   ") == []

def test_remove_stopwords():
    """Test the remove_stopwords function."""
    tokens = ['this', 'is', 'a', 'test', 'of', 'the', 'stopword', 'remover']
    expected = ['test', 'stopword', 'remover']
    assert remove_stopwords(tokens) == expected
    
    custom_stopwords = ['test']
    expected_custom = ['stopword', 'remover']
    assert remove_stopwords(tokens, custom_stopwords=custom_stopwords) == expected_custom

    assert remove_stopwords([]) == []

def test_flag_antipatterns(sample_df):
    """Test the flag_antipatterns function."""
    df = flag_antipatterns(sample_df)
    
    assert 'is_empty_description' in df.columns
    assert df['is_empty_description'].sum() == 2 # one empty, one None
    assert df.loc[2, 'is_empty_description'] == True
    assert df.loc[3, 'is_empty_description'] == True
    
    assert 'is_generic_name' in df.columns
    assert df['is_generic_name'].sum() == 3 # 'Copy of', 'test_', 'v2'
    assert df.loc[1, 'is_generic_name'] == True
    assert df.loc[2, 'is_generic_name'] == True
    assert df.loc[3, 'is_generic_name'] == True
    assert df.loc[0, 'is_generic_name'] == False

def test_preprocess_data(sample_df):
    """Test the full preprocess_data pipeline."""
    processed_df = preprocess_data(sample_df)
    
    assert 'cleaned_name' in processed_df.columns
    assert 'cleaned_description' in processed_df.columns
    assert 'combined_text' in processed_df.columns
    assert 'is_empty_description' in processed_df.columns
    assert 'is_generic_name' in processed_df.columns
    
    # Check a specific row
    assert processed_df.loc[0, 'cleaned_name'] == ['sales', 'report', 'q1']
    assert processed_df.loc[0, 'cleaned_description'] == ['report', 'showing', 'q1', 'sales', 'data']
    assert processed_df.loc[0, 'combined_text'] == 'sales report q1 report showing q1 sales data'
    assert processed_df.loc[0, 'is_empty_description'] == False
    assert processed_df.loc[0, 'is_generic_name'] == False

    # Check empty/None description handling
    assert processed_df.loc[2, 'cleaned_description'] == []
    assert processed_df.loc[3, 'cleaned_description'] == []
    assert processed_df.loc[2, 'combined_text'] == 'test_dashboard' # Only from name

def test_preprocess_data_empty_df():
    """Test the preprocess_data function with an empty DataFrame."""
    empty_df = pd.DataFrame()
    processed_df = preprocess_data(empty_df)
    assert processed_df.empty 