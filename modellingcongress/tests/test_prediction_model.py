import pytest
import numpy as np
import pandas as pd
import torch
from prediction_model import ActionDataset

class TestActionDataset:
  
  @pytest.fixture
  def sample_dataframe(self):
    """Create a sample dataframe for testing ActionDataset"""
    data = {
      'recent_generics': [np.array([1, 2, 3]), np.array([4, 5, 6])],
      'cumulative_generics': [np.array([0.1, 0.2, 0.3]), np.array([0.3, 0.4, 0.5])],
      'recent_categories': [np.array([1, 0, 1]), np.array([0, 1, 0])],
      'cumulative_categories': [np.array([0.5, 0.6, 0.7]), np.array([0.7, 0.8, 0.9])],
      'term': [np.array([1, 0]), np.array([0, 1])],
      'chamber': [np.array([1]), np.array([0])],
      'output_generic': [np.array([0, 1, 0]), np.array([1, 0, 1])],
      'output_categories': [np.array([1, 0, 1]), np.array([0, 1, 0])]
    }
    return pd.DataFrame(data)
  
  def test_dataset_initialization(self, sample_dataframe):
    """Test that ActionDataset initializes correctly"""
    dataset = ActionDataset(sample_dataframe)
    assert len(dataset) == 2
    assert dataset.GENERICS_LEN == 3
    assert dataset.CATEGORIES_LEN == 2
  
  def test_input_concatenation(self, sample_dataframe):
    """Test that inputs are concatenated correctly"""
    dataset = ActionDataset(sample_dataframe)
    expected_input_0 = np.concatenate([
      np.array([1, 2, 3]),  # recent_generics
      np.array([0.1, 0.2, 0.3]),  # cumulative_generics
      np.array([1, 0, 1]),  # recent_categories
      np.array([0.5, 0.6, 0.7]),  # cumulative_categories
      np.array([1, 0]),  # term
      np.array([1])  # chamber
    ])
    np.testing.assert_array_equal(dataset.inputs[0], expected_input_0)
  
  def test_output_concatenation(self, sample_dataframe):
    """Test that outputs are concatenated correctly"""
    dataset = ActionDataset(sample_dataframe)
    expected_output_0 = np.concatenate([
      np.array([0, 1, 0]),  # output_generic
      np.array([1, 0, 1])  # output_categories
    ])
    np.testing.assert_array_equal(dataset.outputs[0], expected_output_0)
  
  def test_getitem(self, sample_dataframe):
    """Test __getitem__ method"""
    dataset = ActionDataset(sample_dataframe)
    input_item, output_item = dataset[0]
    assert isinstance(input_item, np.ndarray)
    assert isinstance(output_item, np.ndarray)
    assert len(input_item) == dataset.input_len()
    assert len(output_item) == dataset.output_len()
  
  def test_length_methods(self, sample_dataframe):
    """Test various length methods"""
    dataset = ActionDataset(sample_dataframe)
    assert dataset.input_len() == 15  # 3+3+3+3+2+1
    assert dataset.output_len() == 6  # 3+3
    assert dataset.generics_len() == 3
    assert dataset.categories_len() == 3
  
  def test_dataset_with_different_sizes(self):
    """Test dataset with different array sizes"""
    data = {
      'recent_generics': [np.array([1, 2]), np.array([3, 4])],
      'cumulative_generics': [np.array([0.1,3]), np.array([0.2,4])],
      'recent_categories': [np.array([1]), np.array([0])],
      'cumulative_categories': [np.array([0.5]), np.array([0.6])],
      'term': [np.array([1]), np.array([0])],
      'chamber': [np.array([1]), np.array([0])],
      'output_generic': [np.array([0, 1]), np.array([1, 0])],
      'output_categories': [np.array([1]), np.array([0])]
    }
    df = pd.DataFrame(data)
    dataset = ActionDataset(df)
    assert dataset.input_len() == 8  # 2+2+1+1+1+1
    assert dataset.output_len() == 3  # 2+1
    assert dataset.generics_len() == 2
    assert dataset.categories_len() == 1