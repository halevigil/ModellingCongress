import pytest
import pandas as pd
import torch
from inference import predict_bill

def test_predict_bill():
  # Create sample test data
  test_data = {
    'action': ['test action 1', 'test action 2'],
    'generic': [None, None],
    'categories': [None, None]
  }
  test_df = pd.DataFrame(test_data)
  
  # Test prediction
  predictions = predict_bill(test_df, refine_first=True)
  
  # Basic validation of output
  assert isinstance(predictions, list)
  assert len(predictions) > 0
  assert isinstance(predictions[0], torch.Tensor)

def test_predict_bill_no_refine():
  # Test with refine_first=False
  test_data = {
    'action': ['test action 1', 'test action 2'],
    'generic': ['generic1', 'generic2'],
    'categories': ['cat1', 'cat2']
  }
  test_df = pd.DataFrame(test_data)
  
  predictions = predict_bill(test_df, refine_first=False)
  
  assert isinstance(predictions, list)
  assert len(predictions) > 0
  assert isinstance(predictions[0], torch.Tensor)

def test_predict_bill_empty_df():
  # Test with empty dataframe
  test_df = pd.DataFrame({'action': [], 'generic': [], 'categories': []})
  
  predictions = predict_bill(test_df)
  
  assert isinstance(predictions, list)
  assert len(predictions) == 0

def test_predict_bill_input_validation():
  # Test with invalid input
  with pytest.raises(Exception):
    predict_bill(None)
    
  with pytest.raises(Exception):
    predict_bill("invalid input")