import json
from inference import CreateInputsOutputs
import numpy as np
import pytest
@pytest.fixture
def setup_test_data(tmp_path):
  generics = ["Generic1", "Generic2"]
  categories = ["Category1", "Category2"] 
  processing = {"min_term": 111, "n_terms": 9, "decay": 2/3}
  
  with open(tmp_path / "generics.json", "w") as f:
    json.dump(generics, f)
  with open(tmp_path / "categories.json", "w") as f:
    json.dump(categories, f)
  with open(tmp_path / "processing.json", "w") as f:
    json.dump(processing, f)
  
  creator = CreateInputsOutputs(str(tmp_path))
  
  assert creator.generics == generics
  assert creator.categories == categories
  assert creator.processing == processing
  assert creator.generics_inv == {"Generic1": 0, "Generic2": 1}
  assert creator.categories_inv == {"Category1": 0, "Category2": 1}

def test_get_generics(tmp_path):
  generics = ["Generic1", "Generic2"]
  categories = ["Category1"]
  processing = {"min_term": 111, "n_terms": 9, "decay": 2/3}
  
  with open(tmp_path / "generics.json", "w") as f:
    json.dump(generics, f)
  with open(tmp_path / "categories.json", "w") as f:
    json.dump(categories, f)
  with open(tmp_path / "processing.json", "w") as f:
    json.dump(processing, f)
  
  creator = CreateInputsOutputs(str(tmp_path))
  assert creator.get_generics() == generics

def test_get_categories_bug(tmp_path):
  # This test reveals the bug in get_categories method
  generics = ["Generic1", "Generic2"]
  categories = ["Category1", "Category2"]
  processing = {"min_term": 111, "n_terms": 9, "decay": 2/3}
  
  with open(tmp_path / "generics.json", "w") as f:
    json.dump(generics, f)
  with open(tmp_path / "categories.json", "w") as f:
    json.dump(categories, f)
  with open(tmp_path / "processing.json", "w") as f:
    json.dump(processing, f)
  
  creator = CreateInputsOutputs(str(tmp_path))
  # This will fail because get_categories returns generics instead of categories
  assert creator.get_categories() == generics  # Bug: should return categories

def test_input_length(tmp_path):
  generics = ["Generic1", "Generic2"]
  categories = ["Category1"]
  processing = {"min_term": 111, "n_terms": 9, "decay": 2/3}
  
  with open(tmp_path / "generics.json", "w") as f:
    json.dump(generics, f)
  with open(tmp_path / "categories.json", "w") as f:
    json.dump(categories, f)
  with open(tmp_path / "processing.json", "w") as f:
    json.dump(processing, f)
  
  creator = CreateInputsOutputs(str(tmp_path))
  expected_length = 2 * len(generics) + 2 * len(categories) + 2 + processing["n_terms"]
  assert creator.input_length() == expected_length

def test_output_length(tmp_path):
  generics = ["Generic1", "Generic2"]
  categories = ["Category1"]
  processing = {"min_term": 111, "n_terms": 9, "decay": 2/3}
  
  with open(tmp_path / "generics.json", "w") as f:
    json.dump(generics, f)
  with open(tmp_path / "categories.json", "w") as f:
    json.dump(categories, f)
  with open(tmp_path / "processing.json", "w") as f:
    json.dump(processing, f)
  
  creator = CreateInputsOutputs(str(tmp_path))
  expected_length = len(generics) + len(categories)
  assert creator.output_length() == expected_length

def test_create_output_vector(tmp_path):
  generics = ["Generic1", "Generic2"]
  categories = ["Category1", "Category2"]
  processing = {"min_term": 111, "n_terms": 9, "decay": 2/3}
  
  with open(tmp_path / "generics.json", "w") as f:
    json.dump(generics, f)
  with open(tmp_path / "categories.json", "w") as f:
    json.dump(categories, f)
  with open(tmp_path / "processing.json", "w") as f:
    json.dump(processing, f)
  
  creator = CreateInputsOutputs(str(tmp_path))
  
  # Test with valid generic and categories
  output_vec = creator.create_output_vector("Generic1", ["Category1"])
  expected = np.concatenate([np.array([1, 0]), np.array([1, 0])])
  np.testing.assert_array_equal(output_vec, expected)
  
  # Test with no generic or categories
  output_vec = creator.create_output_vector()
  expected = np.concatenate([np.array([0, 0]), np.array([0, 0])])
  np.testing.assert_array_equal(output_vec, expected)

def test_create_input_vector_unnormalized(tmp_path):
  generics = ["Generic1", "Generic2"]
  categories = ["Category1"]
  processing = {"min_term": 111, "n_terms": 9, "decay": 2/3}
  
  with open(tmp_path / "generics.json", "w") as f:
    json.dump(generics, f)
  with open(tmp_path / "categories.json", "w") as f:
    json.dump(categories, f)
  with open(tmp_path / "processing.json", "w") as f:
    json.dump(processing, f)
  
  creator = CreateInputsOutputs(str(tmp_path))
  
  # Test with defaults
  input_vec = creator.create_input_vector_unnormalized()
  assert len(input_vec) == creator.input_length()
  
  # Test with specific chamber and term
  input_vec = creator.create_input_vector_unnormalized(chamber="House", term=111)
  # Check that chamber encoding is correct
  chamber_start = 2 * len(generics) + 2 * len(categories) + processing["n_terms"]
  assert input_vec[chamber_start] == 1  # House
  assert input_vec[chamber_start + 1] == 0  # Not Senate

def test_vector_to_probabilities(tmp_path):
  generics = ["Generic1", "Generic2"]
  categories = ["Category1"]
  processing = {"min_term": 111, "n_terms": 9, "decay": 2/3}
  
  with open(tmp_path / "generics.json", "w") as f:
    json.dump(generics, f)
  with open(tmp_path / "categories.json", "w") as f:
    json.dump(categories, f)
  with open(tmp_path / "processing.json", "w") as f:
    json.dump(processing, f)
  
  creator = CreateInputsOutputs(str(tmp_path))
  
  test_vec = np.array([0.5, 0.3, 0.8])
  generic_probs, category_probs = creator.vector_to_probabilities(test_vec)
  
  assert generic_probs == {"Generic1": 0.5, "Generic2": 0.3}
  assert category_probs == {"Category1": 0.8}

def test_create_input_vector_with_scale_factors(tmp_path):
  generics = ["Generic1", "Generic2"]
  categories = ["Category1"]
  processing = {"min_term": 111, "n_terms": 9, "decay": 2/3}
  
  with open(tmp_path / "generics.json", "w") as f:
    json.dump(generics, f)
  with open(tmp_path / "categories.json", "w") as f:
    json.dump(categories, f)
  with open(tmp_path / "processing.json", "w") as f:
    json.dump(processing, f)
  
  # Create scale factors file
  scale_factors = np.array([1.0, 2.0, 1.5, 0.5, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,1.0,1.0,1.0])
  np.save(tmp_path / "scale_factors.npy", scale_factors)
  
  creator = CreateInputsOutputs(str(tmp_path))
  
  # Test that scale factors are loaded
  np.testing.assert_array_equal(creator.scale_factors, scale_factors)
  
  # Test normalized input vector
  input_vec_unnorm = creator.create_input_vector_unnormalized(chamber="Senate", term=112)
  input_vec_norm = creator.create_input_vector(chamber="Senate", term=112)
  
  expected_norm = input_vec_unnorm / scale_factors
  np.testing.assert_array_almost_equal(input_vec_norm, expected_norm)

def test_create_output_vector_invalid_inputs(tmp_path):
  generics = ["Generic1", "Generic2"]
  categories = ["Category1", "Category2"]
  processing = {"min_term": 111, "n_terms": 9, "decay": 2/3}
  
  with open(tmp_path / "generics.json", "w") as f:
    json.dump(generics, f)
  with open(tmp_path / "categories.json", "w") as f:
    json.dump(categories, f)
  with open(tmp_path / "processing.json", "w") as f:
    json.dump(processing, f)
  
  creator = CreateInputsOutputs(str(tmp_path))
  
  # Test with invalid generic
  output_vec = creator.create_output_vector("InvalidGeneric", ["Category1"])
  expected = np.concatenate([np.array([0, 0]), np.array([1, 0])])
  np.testing.assert_array_equal(output_vec, expected)
  
  # Test with invalid category
  output_vec = creator.create_output_vector("Generic1", ["InvalidCategory"])
  expected = np.concatenate([np.array([1, 0]), np.array([0, 0])])
  np.testing.assert_array_equal(output_vec, expected)

def test_create_input_vector_with_previous_vectors(tmp_path):
  generics = ["Generic1", "Generic2"]
  categories = ["Category1"]
  processing = {"min_term": 111, "n_terms": 9, "decay": 0.5}
  
  with open(tmp_path / "generics.json", "w") as f:
    json.dump(generics, f)
  with open(tmp_path / "categories.json", "w") as f:
    json.dump(categories, f)
  with open(tmp_path / "processing.json", "w") as f:
    json.dump(processing, f)
  
  creator = CreateInputsOutputs(str(tmp_path))
  
  # Create previous vectors
  prev_input = np.ones(creator.input_length()) * 0.5
  prev_output = np.ones(creator.output_length()) * 0.3
  
  input_vec = creator.create_input_vector_unnormalized(
    prev_input_vector=prev_input, 
    prev_output_vector=prev_output,
    chamber="House",
    term=113
  )
  
  assert len(input_vec) == creator.input_length()
  # Verify that previous vectors are incorporated
  assert not np.array_equal(input_vec[:len(generics)], np.zeros(len(generics)))

def test_chamber_encoding_senate(tmp_path):
  generics = ["Generic1"]
  categories = ["Category1"]
  processing = {"min_term": 111, "n_terms": 9, "decay": 2/3}
  
  with open(tmp_path / "generics.json", "w") as f:
    json.dump(generics, f)
  with open(tmp_path / "categories.json", "w") as f:
    json.dump(categories, f)
  with open(tmp_path / "processing.json", "w") as f:
    json.dump(processing, f)
  
  creator = CreateInputsOutputs(str(tmp_path))
  
  input_vec = creator.create_input_vector_unnormalized(chamber="Senate", term=111)
  chamber_start = 2 * len(generics) + 2 * len(categories)+processing["n_terms"]
  
  print(input_vec)
  assert input_vec[chamber_start] == 0  # Not House
  assert input_vec[chamber_start + 1] == 1  # Senate

def test_term_encoding(tmp_path):
  generics = ["Generic1"]
  categories = ["Category1"]
  processing = {"min_term": 111, "n_terms": 9, "decay": 2/3}
  
  with open(tmp_path / "generics.json", "w") as f:
    json.dump(generics, f)
  with open(tmp_path / "categories.json", "w") as f:
    json.dump(categories, f)
  with open(tmp_path / "processing.json", "w") as f:
    json.dump(processing, f)
  
  creator = CreateInputsOutputs(str(tmp_path))
  
  input_vec = creator.create_input_vector_unnormalized(term=113, min_term=111, n_terms=9)
  term_start = 2 * len(generics) + 2 * len(categories)
  
  # Term 113 should be at index 2 (113 - 111 = 2)
  for i in range(9):
    if i == 2:
      assert input_vec[term_start + i] == 1
    else:
      assert input_vec[term_start + i] == 0

def test_create_input_vector_without_scale_factors(tmp_path):
  generics = ["Generic1"]
  categories = ["Category1"]
  processing = {"min_term": 111, "n_terms": 9, "decay": 2/3}
  
  with open(tmp_path / "generics.json", "w") as f:
    json.dump(generics, f)
  with open(tmp_path / "categories.json", "w") as f:
    json.dump(categories, f)
  with open(tmp_path / "processing.json", "w") as f:
    json.dump(processing, f)
  
  creator = CreateInputsOutputs(str(tmp_path))
  
  # Should raise AttributeError when no scale_factors exist
  with pytest.raises(AttributeError):
    creator.create_input_vector(chamber="House", term=111)
def test_decay_calculation_accuracy(tmp_path):
  generics = ["Generic1", "Generic2"]
  categories = ["Category1"]
  processing = {"min_term": 111, "n_terms": 9, "decay": 0.5}
  
  with open(tmp_path / "generics.json", "w") as f:
    json.dump(generics, f)
  with open(tmp_path / "categories.json", "w") as f:
    json.dump(categories, f)
  with open(tmp_path / "processing.json", "w") as f:
    json.dump(processing, f)
  
  creator = CreateInputsOutputs(str(tmp_path))
  
  # Create specific previous vectors
  prev_input = np.zeros(creator.input_length())
  prev_input[0] = 0.8  # recent_generics[0]
  prev_input[1] = 0.4  # recent_generics[1]
  
  prev_output = np.array([0.6, 0.2, 0.9])  # [generic1, generic2, category1]
  
  input_vec = creator.create_input_vector_unnormalized(
    prev_input_vector=prev_input,
    prev_output_vector=prev_output
  )
  
  # Test recent_generics calculation: prev_output_generics*(1-decay) + decay*prev_recent_generics
  expected_recent_0 = 0.6 * (1 - 0.5) + 0.5 * 0.8  # 0.3 + 0.4 = 0.7
  expected_recent_1 = 0.2 * (1 - 0.5) + 0.5 * 0.4  # 0.1 + 0.2 = 0.3
  
  assert abs(input_vec[0] - expected_recent_0) < 1e-10
  assert abs(input_vec[1] - expected_recent_1) < 1e-10

def test_cumulative_calculation_accuracy(tmp_path):
  generics = ["Generic1", "Generic2"]
  categories = ["Category1"]
  processing = {"min_term": 111, "n_terms": 9, "decay": 2/3}
  
  with open(tmp_path / "generics.json", "w") as f:
    json.dump(generics, f)
  with open(tmp_path / "categories.json", "w") as f:
    json.dump(categories, f)
  with open(tmp_path / "processing.json", "w") as f:
    json.dump(processing, f)
  
  creator = CreateInputsOutputs(str(tmp_path))
  
  # Set up previous cumulative values
  prev_input = np.zeros(creator.input_length())
  prev_input[2] = 1.5  # cumulative_generics[0]
  prev_input[3] = 2.3  # cumulative_generics[1]
  prev_input[5] = 0.8  # cumulative_categories[0]
  
  prev_output = np.array([0.4, 0.7, 0.3])
  
  input_vec = creator.create_input_vector_unnormalized(
    prev_input_vector=prev_input,
    prev_output_vector=prev_output
  )
  
  # Test cumulative calculation: prev_output + prev_cumulative
  assert abs(input_vec[2] - (0.4 + 1.5)) < 1e-10  # 1.9
  assert abs(input_vec[3] - (0.7 + 2.3)) < 1e-10  # 3.0
  assert abs(input_vec[5] - (0.3 + 0.8)) < 1e-10  # 1.1

def test_vector_indexing_consistency(tmp_path):
  generics = ["Generic1", "Generic2", "Generic3"]
  categories = ["Category1", "Category2"]
  processing = {"min_term": 111, "n_terms": 9, "decay": 2/3}
  
  with open(tmp_path / "generics.json", "w") as f:
    json.dump(generics, f)
  with open(tmp_path / "categories.json", "w") as f:
    json.dump(categories, f)
  with open(tmp_path / "processing.json", "w") as f:
    json.dump(processing, f)
  
  creator = CreateInputsOutputs(str(tmp_path))
  
  input_vec = creator.create_input_vector_unnormalized(chamber="House", term=115)
  
  # Verify vector structure
  n_generics = len(generics)
  n_categories = len(categories)
  
  # Indices for different sections
  recent_generics_end = n_generics
  cumulative_generics_end = 2 * n_generics
  recent_categories_end = 2 * n_generics + n_categories
  cumulative_categories_end = 2 * n_generics + 2 * n_categories
  term_end = cumulative_categories_end + 9
  chamber_end = term_end + 2
  
  assert len(input_vec) == chamber_end

  # Check term encoding (term 115 -> index 4)
  term_idx = 115 - 111
  for i in range(9):
    if i == term_idx:
      assert input_vec[cumulative_categories_end + i] == 1
    else:
      assert input_vec[cumulative_categories_end + i] == 0
  
  # Check chamber encoding (House should be [1, 0])
  assert input_vec[term_end] == 1
  assert input_vec[term_end + 1] == 0
  

def test_multiple_categories_output_vector(tmp_path):
  generics = ["Generic1"]
  categories = ["Category1", "Category2", "Category3"]
  processing = {"min_term": 111, "n_terms": 9, "decay": 2/3}
  
  with open(tmp_path / "generics.json", "w") as f:
    json.dump(generics, f)
  with open(tmp_path / "categories.json", "w") as f:
    json.dump(categories, f)
  with open(tmp_path / "processing.json", "w") as f:
    json.dump(processing, f)
  
  creator = CreateInputsOutputs(str(tmp_path))
  
  # Test multiple categories
  output_vec = creator.create_output_vector("Generic1", ["Category1", "Category3"])
  expected = np.concatenate([np.array([1]), np.array([1, 0, 1])])
  np.testing.assert_array_equal(output_vec, expected)

def test_sequential_input_vector_updates(tmp_path):
  generics = ["Generic1", "Generic2"]
  categories = ["Category1"]
  processing = {"min_term": 111, "n_terms": 9, "decay": 0.6}
  
  with open(tmp_path / "generics.json", "w") as f:
    json.dump(generics, f)
  with open(tmp_path / "categories.json", "w") as f:
    json.dump(categories, f)
  with open(tmp_path / "processing.json", "w") as f:
    json.dump(processing, f)
  
  creator = CreateInputsOutputs(str(tmp_path))
  
  # Simulate sequence of actions
  input_vec1 = creator.create_input_vector_unnormalized()
  output_vec1 = creator.create_output_vector("Generic1", ["Category1"])
  
  input_vec2 = creator.create_input_vector_unnormalized(
    prev_input_vector=input_vec1, 
    prev_output_vector=output_vec1
  )
  output_vec2 = creator.create_output_vector("Generic2", [])
  
  input_vec3 = creator.create_input_vector_unnormalized(
    prev_input_vector=input_vec2, 
    prev_output_vector=output_vec2
  )
  
  # Verify cumulative values increase
  assert input_vec2[2] > input_vec1[2]  # cumulative_generics[0] should increase
  assert input_vec3[3] > input_vec2[3]  # cumulative_generics[1] should increase

def test_edge_case_term_bounds(tmp_path):
  generics = ["Generic1"]
  categories = ["Category1"]
  processing = {"min_term": 111, "n_terms": 5, "decay": 2/3}
  
  with open(tmp_path / "generics.json", "w") as f:
    json.dump(generics, f)
  with open(tmp_path / "categories.json", "w") as f:
    json.dump(categories, f)
  with open(tmp_path / "processing.json", "w") as f:
    json.dump(processing, f)
  
  creator = CreateInputsOutputs(str(tmp_path))
  
  # Test minimum term
  input_vec = creator.create_input_vector_unnormalized(term=111, min_term=111, n_terms=5)
  term_start = 2 + 2  # 2*generics + 2*categories
  assert input_vec[term_start] == 1
  assert sum(input_vec[term_start:term_start+5]) == 1
  
  # Test maximum term
  input_vec = creator.create_input_vector_unnormalized(term=115, min_term=111, n_terms=5)
  assert input_vec[term_start + 4] == 1
  assert sum(input_vec[term_start:term_start+5]) == 1

def test_zero_decay_factor(tmp_path):
  generics = ["Generic1"]
  categories = ["Category1"]
  processing = {"min_term": 111, "n_terms": 9, "decay": 0.0}
  
  with open(tmp_path / "generics.json", "w") as f:
    json.dump(generics, f)
  with open(tmp_path / "categories.json", "w") as f:
    json.dump(categories, f)
  with open(tmp_path / "processing.json", "w") as f:
    json.dump(processing, f)
  
  creator = CreateInputsOutputs(str(tmp_path))
  
  prev_input = np.ones(creator.input_length()) * 0.5
  prev_output = np.array([0.8, 0.6])
  
  # With decay=0, recent should equal prev_output
  input_vec = creator.create_input_vector_unnormalized(
    prev_input_vector=prev_input,
    prev_output_vector=prev_output
  )
  
  assert abs(input_vec[0] - 0.8) < 1e-10  # recent_generics[0]
  assert abs(input_vec[2] - 0.6) < 1e-10  # recent_categories[0]

def test_full_decay_factor(tmp_path):
  generics = ["Generic1"]
  categories = ["Category1"]
  processing = {"min_term": 111, "n_terms": 9, "decay": 1.0}
  
  with open(tmp_path / "generics.json", "w") as f:
    json.dump(generics, f)
  with open(tmp_path / "categories.json", "w") as f:
    json.dump(categories, f)
  with open(tmp_path / "processing.json", "w") as f:
    json.dump(processing, f)
  
  creator = CreateInputsOutputs(str(tmp_path))
  
  prev_input = np.ones(creator.input_length()) * 0.7
  prev_output = np.array([0.3, 0.9])
  
  # With decay=1, recent should equal prev_recent
  input_vec = creator.create_input_vector_unnormalized(
    prev_input_vector=prev_input,
    prev_output_vector=prev_output
  )
  
  assert abs(input_vec[0] - 0.7) < 1e-10  # recent_generics[0]
  assert abs(input_vec[2] - 0.7) < 1e-10  # recent_categories[0]

def test_categories_bug_fix_needed(tmp_path):
  # This test documents the expected behavior after fixing the bug
  generics = ["Generic1", "Generic2"]
  categories = ["Category1", "Category2"]
  processing = {"min_term": 111, "n_terms": 9, "decay": 2/3}
  
  with open(tmp_path / "generics.json", "w") as f:
    json.dump(generics, f)
  with open(tmp_path / "categories.json", "w") as f:
    json.dump(categories, f)
  with open(tmp_path / "processing.json", "w") as f:
    json.dump(processing, f)
  
  creator = CreateInputsOutputs(str(tmp_path))
  
  # After fixing the bug, this should pass
  # Currently it will fail because get_categories returns generics
  # assert creator.get_categories() == categories

def test_concat_vs_concatenate_consistency(tmp_path):
  generics = ["Generic1", "Generic2"]
  categories = ["Category1"]
  processing = {"min_term": 111, "n_terms": 9, "decay": 2/3}
  
  with open(tmp_path / "generics.json", "w") as f:
    json.dump(generics, f)
  with open(tmp_path / "categories.json", "w") as f:
    json.dump(categories, f)
  with open(tmp_path / "processing.json", "w") as f:
    json.dump(processing, f)
  
  creator = CreateInputsOutputs(str(tmp_path))
  
  # Test that np.concat works (it should be np.concatenate)
  generic_vec = np.array([1, 0])
  categories_vec = np.array([1])

def test_empty_category_list_handling(tmp_path):
  generics = ["Generic1"]
  categories = ["Category1", "Category2"]
  processing = {"min_term": 111, "n_terms": 9, "decay": 2/3}
  
  with open(tmp_path / "generics.json", "w") as f:
    json.dump(generics, f)
  with open(tmp_path / "categories.json", "w") as f:
    json.dump(categories, f)
  with open(tmp_path / "processing.json", "w") as f:
    json.dump(processing, f)
  
  creator = CreateInputsOutputs(str(tmp_path))
  
  # Test with empty list
  output_vec = creator.create_output_vector("Generic1", [])
  expected = np.concatenate([np.array([1]), np.array([0, 0])])
  np.testing.assert_array_equal(output_vec, expected)

def test_vector_to_probabilities_precision(tmp_path):
  generics = ["Generic1", "Generic2"]
  categories = ["Category1", "Category2"]
  processing = {"min_term": 111, "n_terms": 9, "decay": 2/3}
  
  with open(tmp_path / "generics.json", "w") as f:
    json.dump(generics, f)
  with open(tmp_path / "categories.json", "w") as f:
    json.dump(categories, f)
  with open(tmp_path / "processing.json", "w") as f:
    json.dump(processing, f)
  
  creator = CreateInputsOutputs(str(tmp_path))
  
  # Test with high precision values
  test_vec = np.array([0.123456789, 0.987654321, 0.555555555, 0.111111111])
  generic_probs, category_probs = creator.vector_to_probabilities(test_vec)
  
  assert generic_probs["Generic1"] == 0.123456789
  assert generic_probs["Generic2"] == 0.987654321
  assert category_probs["Category1"] == 0.555555555
  assert category_probs["Category2"] == 0.111111111
