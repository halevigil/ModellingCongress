import pytest
import pandas as pd
import numpy as np
from modellingcongress.prepare_inputs_outputs import (
  create_next_vector,
  create_next_vector_unnormalized,
  create_vectors_bill_unnormalized,
  normalize
)


def test_create_next_vector_unnormalized_initial():
  """Test create_next_vector_unnormalized with no previous vectors."""
  all_generics = ["generic1", "generic2", "generic3"]
  all_categories = ["cat1", "cat2"]
  
  result = create_next_vector_unnormalized(
    decay=0.5,
    all_generics=all_generics,
    all_categories=all_categories,
    curr_generic="generic1",
    curr_categories=["cat1"],
    term=111,
    chamber="House"
  )
  
  assert result["output_generic"][0] == 1
  assert result["output_generic"][1] == 0
  assert result["output_categories"][0] == 1
  assert result["chamber"][0] == 1
  assert result["chamber"][1] == 0
  assert result["term"][0] == 1
  assert len(result["recent_generics"]) == 3
  assert len(result["cum_prev_generics"]) == 3


def test_create_next_vector_unnormalized_with_previous():
  """Test create_next_vector_unnormalized with previous vectors."""
  all_generics = ["generic1", "generic2"]
  all_categories = ["cat1", "cat2"]
  prev_vectors = {
    "cum_prev_generics": np.array([1, 0]),
    "recent_generics": np.array([0.5, 0]),
    "output_generic": np.array([1, 0]),
    "cum_prev_categories": np.array([1, 0]),
    "recent_categories": np.array([0.5, 0]),
    "output_categories": np.array([1, 0])
  }
  
def test_create_next_vector_unnormalized_decay_calculation():
  """Test that decay calculation works correctly for recent vectors."""
  all_generics = ["generic1", "generic2"]
  all_categories = ["cat1", "cat2"]
  decay = 0.3
  
  prev_vectors = {
    "cum_prev_generics": np.array([2, 1]),
    "recent_generics": np.array([0.8, 0.4]),
    "output_generic": np.array([1, 0]),
    "cum_prev_categories": np.array([1, 2]),
    "recent_categories": np.array([0.6, 0.2]),
    "output_categories": np.array([1, 0])
  }
  
  result = create_next_vector_unnormalized(
    decay=decay,
    all_generics=all_generics,
    all_categories=all_categories,
    curr_generic="generic2",
    curr_categories=["cat2"],
    prev_vectors=prev_vectors,
    term=112,
    chamber="Senate"
  )
  
  # Test recent_generics: prev_output * (1-decay) + decay * prev_recent
  expected_recent_generics = np.array([1, 0]) * (1 - decay) + decay * np.array([0.8, 0.4])
  np.testing.assert_array_almost_equal(result["recent_generics"], expected_recent_generics)
  
  # Test recent_categories
  expected_recent_categories = np.array([1, 0]) * (1 - decay) + decay * np.array([0.6, 0.2])
  np.testing.assert_array_almost_equal(result["recent_categories"], expected_recent_categories)


def test_create_next_vector_unnormalized_cumulative_calculation():
  """Test cumulative vector calculation."""
  all_generics = ["generic1", "generic2", "generic3"]
  all_categories = ["cat1", "cat2"]
  
  prev_vectors = {
    "cum_prev_generics": np.array([3, 1, 2]),
    "recent_generics": np.array([0.5, 0.2, 0.1]),
    "output_generic": np.array([0, 1, 0]),
    "cum_prev_categories": np.array([2, 1]),
    "recent_categories": np.array([0.3, 0.7]),
    "output_categories": np.array([0, 1])
  }
  
  result = create_next_vector_unnormalized(
    decay=0.5,
    all_generics=all_generics,
    all_categories=all_categories,
    curr_generic="generic1",
    curr_categories=["cat1"],
    prev_vectors=prev_vectors,
    term=113,
    chamber="House"
  )
  
  # Test cum_prev_generics: prev_output + prev_cumulative
  expected_cum_generics = np.array([0, 1, 0]) + np.array([3, 1, 2])
  np.testing.assert_array_equal(result["cum_prev_generics"], expected_cum_generics)
  
  # Test cum_prev_categories
  expected_cum_categories = np.array([0, 1]) + np.array([2, 1])
  np.testing.assert_array_equal(result["cum_prev_categories"], expected_cum_categories)


def test_create_next_vector_unnormalized_multiple_categories():
  """Test with multiple categories for current action."""
  all_generics = ["generic1", "generic2"]
  all_categories = ["cat1", "cat2", "cat3", "cat4"]
  
  result = create_next_vector_unnormalized(
    decay=0.5,
    all_generics=all_generics,
    all_categories=all_categories,
    curr_generic="generic1",
    curr_categories=["cat1", "cat3", "cat4"],
    term=111,
    chamber="House"
  )
  
  expected_categories = np.array([1, 0, 1, 1])
  np.testing.assert_array_equal(result["output_categories"], expected_categories)


def test_create_next_vector_unnormalized_no_current_data():
  """Test with no current generic or categories."""
  all_generics = ["generic1", "generic2"]
  all_categories = ["cat1", "cat2"]
  
  result = create_next_vector_unnormalized(
    decay=0.5,
    all_generics=all_generics,
    all_categories=all_categories,
    curr_generic=None,
    curr_categories=None,
    term=111,
    chamber="House"
  )
  
  np.testing.assert_array_equal(result["output_generic"], np.zeros(2))
  np.testing.assert_array_equal(result["output_categories"], np.zeros(2))


def test_create_next_vector_unnormalized_term_bounds():
  """Test term vector with different term values."""
  all_generics = ["generic1"]
  all_categories = ["cat1"]
  min_term = 111
  n_terms = 5
  
  # Test first term
  result1 = create_next_vector_unnormalized(
    decay=0.5,
    all_generics=all_generics,
    all_categories=all_categories,
    term=111,
    min_term=min_term,
    n_terms=n_terms
  )
  expected_term1 = np.array([1, 0, 0, 0, 0])
  np.testing.assert_array_equal(result1["term"], expected_term1)
  
  # Test middle term
  result2 = create_next_vector_unnormalized(
    decay=0.5,
    all_generics=all_generics,
    all_categories=all_categories,
    term=113,
    min_term=min_term,
    n_terms=n_terms
  )
  expected_term2 = np.array([0, 0, 1, 0, 0])
  np.testing.assert_array_equal(result2["term"], expected_term2)
  
  # Test last term
  result3 = create_next_vector_unnormalized(
    decay=0.5,
    all_generics=all_generics,
    all_categories=all_categories,
    term=115,
    min_term=min_term,
    n_terms=n_terms
  )
  expected_term3 = np.array([0, 0, 0, 0, 1])
  np.testing.assert_array_equal(result3["term"], expected_term3)


def test_create_next_vector_unnormalized_chamber_encoding():
  """Test chamber one-hot encoding."""
  all_generics = ["generic1"]
  all_categories = ["cat1"]
  
  # Test House
  result_house = create_next_vector_unnormalized(
    decay=0.5,
    all_generics=all_generics,
    all_categories=all_categories,
    chamber="House"
  )
  np.testing.assert_array_equal(result_house["chamber"], np.array([1, 0]))
  
  # Test Senate
  result_senate = create_next_vector_unnormalized(
    decay=0.5,
    all_generics=all_generics,
    all_categories=all_categories,
    chamber="Senate"
  )
  np.testing.assert_array_equal(result_senate["chamber"], np.array([0, 1]))
  
  # Test invalid chamber
  result_invalid = create_next_vector_unnormalized(
    decay=0.5,
    all_generics=all_generics,
    all_categories=all_categories,
    chamber="InvalidChamber"
  )
  np.testing.assert_array_equal(result_invalid["chamber"], np.array([0, 0]))


def test_create_next_vector_unnormalized_zero_decay():
  """Test with zero decay factor."""
  all_generics = ["generic1", "generic2"]
  all_categories = ["cat1", "cat2"]
  
  prev_vectors = {
    "cum_prev_generics": np.array([1, 2]),
    "recent_generics": np.array([0.5, 0.3]),
    "output_generic": np.array([1, 0]),
    "cum_prev_categories": np.array([1, 1]),
    "recent_categories": np.array([0.4, 0.6]),
    "output_categories": np.array([1, 0])
  }
  
  result = create_next_vector_unnormalized(
    decay=0.0,
    all_generics=all_generics,
    all_categories=all_categories,
    curr_generic="generic2",
    curr_categories=["cat2"],
    prev_vectors=prev_vectors
  )
  
  # With zero decay, recent should equal prev_output * 1 + 0 * prev_recent = prev_output
  np.testing.assert_array_equal(result["recent_generics"], np.array([1, 0]))
  np.testing.assert_array_equal(result["recent_categories"], np.array([1, 0]))


def test_create_next_vector_unnormalized_full_decay():
  """Test with full decay factor (1.0)."""
  all_generics = ["generic1", "generic2"]
  all_categories = ["cat1", "cat2"]
  
  prev_vectors = {
    "cum_prev_generics": np.array([1, 2]),
    "recent_generics": np.array([0.5, 0.3]),
    "output_generic": np.array([1, 0]),
    "cum_prev_categories": np.array([1, 1]),
    "recent_categories": np.array([0.4, 0.6]),
    "output_categories": np.array([1, 0])
  }
  
  result = create_next_vector_unnormalized(
    decay=1.0,
    all_generics=all_generics,
    all_categories=all_categories,
    curr_generic="generic2",
    curr_categories=["cat2"],
    prev_vectors=prev_vectors
  )
  
  # With full decay, recent should equal prev_output * 0 + 1 * prev_recent = prev_recent
  np.testing.assert_array_equal(result["recent_generics"], np.array([0.5, 0.3]))
  np.testing.assert_array_equal(result["recent_categories"], np.array([0.4, 0.6]))
  
  assert result["output_generic"][1] == 1
  assert result["cum_prev_generics"][0] == 2  # 1 + 1
  assert result["chamber"][1] == 1
  assert result["term"][1] == 1


def test_create_next_vector_unnormalized_invalid_generic():
  """Test with invalid generic name."""
  all_generics = ["generic1", "generic2"]
  all_categories = ["cat1", "cat2"]
  
  result = create_next_vector_unnormalized(
    decay=0.5,
    all_generics=all_generics,
    all_categories=all_categories,
    curr_generic="invalid_generic",
    curr_categories=["cat1"]
  )
  
  assert np.sum(result["output_generic"]) == 0


def test_create_vectors_bill_unnormalized():
  """Test create_vectors_bill_unnormalized function."""
  bill_df = pd.DataFrame({
    "generic": ["generic1", "generic2"],
    "categories": [["cat1"], ["cat2"]],
    "term": [111, 111],
    "chamber": ["House", "Senate"]
  })
  all_generics = ["generic1", "generic2"]
  all_categories = ["cat1", "cat2"]
  
  result = create_vectors_bill_unnormalized(bill_df, all_generics, all_categories, decay=0.5)
  
  assert len(result) == 2
  assert "recent_generics" in result.columns
  assert "output_generic" in result.columns


def test_normalize():
  """Test normalize function."""
  to_normalize = {
    "cum_prev_generics": [np.array([2, 4]), np.array([1, 2])],
    "cum_prev_categories": [np.array([1, 3]), np.array([2, 1])]
  }
  std_generics = np.array([1, 2])
  std_categories = np.array([0.5, 1])
  
  result = normalize(to_normalize, std_generics, std_categories)
  
  assert isinstance(result["cum_prev_generics"], list)
  assert len(result["cum_prev_generics"]) == 2


def test_create_next_vector():
  """Test create_next_vector function (normalized version)."""
  all_generics = ["generic1", "generic2"]
  all_categories = ["cat1", "cat2"]
  std_generics = np.array([1, 1])
  std_categories = np.array([1, 1])
  
  result = create_next_vector(
    std_generics=std_generics,
    std_categories=std_categories,
    decay=0.5,
    all_generics=all_generics,
    all_categories=all_categories,
    curr_generic="generic1",
    curr_categories=["cat1"]
  )
  
  assert "recent_generics" in result
  assert "cum_prev_generics" in result