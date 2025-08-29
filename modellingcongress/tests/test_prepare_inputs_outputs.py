import tempfile
import os
import json
import numpy as np
import tempfile
import os
import json
import numpy as np
import tempfile
import os
import json
import numpy as np
import tempfile
import os
import json
import numpy as np
import tempfile
import os
import json
import numpy as np
import tempfile
import os
import json
import numpy as np
import tempfile
import os
import json
import numpy as np
from prepare_inputs_outputs import CreateInputsOutputs

def test_create_input_vector_unnormalized_default():
  """Test create_input_vector_unnormalized with default parameters"""
  
  with tempfile.TemporaryDirectory() as temp_dir:
    # Setup test data
    generics = ["Generic1", "Generic2"]
    categories = ["Category1", "Category2"]
    processing = {"decay": 0.8, "min_term": 111, "n_terms": 9}
    
    with open(os.path.join(temp_dir, "generics.json"), "w") as f:
      json.dump(generics, f)
    with open(os.path.join(temp_dir, "categories.json"), "w") as f:
      json.dump(categories, f)
    with open(os.path.join(temp_dir, "processing.json"), "w") as f:
      json.dump(processing, f)
    
    creator = CreateInputsOutputs(temp_dir)
    result = creator.create_input_vector_unnormalized()
    
    expected_length = 2 * len(generics) + 2 * len(categories) + 2 + 9
    assert len(result) == expected_length
    assert np.all(result == 0)


def test_create_input_vector_unnormalized_with_prev_generic():
  """Test create_input_vector_unnormalized with previous generic"""
  
  with tempfile.TemporaryDirectory() as temp_dir:
    generics = ["Generic1", "Generic2"]
    categories = ["Category1", "Category2"]
    processing = {"decay": 0.8, "min_term": 111, "n_terms": 9}
    
    with open(os.path.join(temp_dir, "generics.json"), "w") as f:
      json.dump(generics, f)
    with open(os.path.join(temp_dir, "categories.json"), "w") as f:
      json.dump(categories, f)
    with open(os.path.join(temp_dir, "processing.json"), "w") as f:
      json.dump(processing, f)
    
    creator = CreateInputsOutputs(temp_dir)
    result = creator.create_input_vector_unnormalized(prev_generic="Generic1")
    
    # Check that the first generic is set to 1
    assert result[0] == 1
    assert result[1] == 0
    # Check that recent generics are also updated
    assert (result[2] - 0.2)<1e-5  # (1-decay) * 1


def test_create_input_vector_unnormalized_with_prev_categories():
  """Test create_input_vector_unnormalized with previous categories"""
  
  with tempfile.TemporaryDirectory() as temp_dir:
    generics = ["Generic1", "Generic2"]
    categories = ["Category1", "Category2"]
    processing = {"decay": 0.8, "min_term": 111, "n_terms": 9}
    
    with open(os.path.join(temp_dir, "generics.json"), "w") as f:
      json.dump(generics, f)
    with open(os.path.join(temp_dir, "categories.json"), "w") as f:
      json.dump(categories, f)
    with open(os.path.join(temp_dir, "processing.json"), "w") as f:
      json.dump(processing, f)
    
    creator = CreateInputsOutputs(temp_dir)
    result = creator.create_input_vector_unnormalized(prev_categories=["Category1", "Category2"])
    
    # Check that categories are set correctly
    assert result[4] == 1  # first category
    assert result[5] == 1  # second category
    # Check recent categories
    assert (result[6]- 0.2)<=1e-5  # (1-decay) * 1
    assert (result[7] - 0.2)<=1e-5  # (1-decay) * 1


def test_create_input_vector_unnormalized_with_chamber():
  """Test create_input_vector_unnormalized with chamber parameter"""
  
  with tempfile.TemporaryDirectory() as temp_dir:
    generics = ["Generic1", "Generic2"]
    categories = ["Category1", "Category2"]
    processing = {"decay": 0.8, "min_term": 111, "n_terms": 9}
    
    with open(os.path.join(temp_dir, "generics.json"), "w") as f:
      json.dump(generics, f)
    with open(os.path.join(temp_dir, "categories.json"), "w") as f:
      json.dump(categories, f)
    with open(os.path.join(temp_dir, "processing.json"), "w") as f:
      json.dump(processing, f)
    
    creator = CreateInputsOutputs(temp_dir)
    
    # Test House
    result_house = creator.create_input_vector_unnormalized(chamber="House")
    assert result_house[-2] == 1  # House chamber bit
    assert result_house[-1] == 0  # Senate chamber bit
    
    # Test Senate
    result_senate = creator.create_input_vector_unnormalized(chamber="Senate")
    assert result_senate[-2] == 0  # House chamber bit
    assert result_senate[-1] == 1  # Senate chamber bit


def test_create_input_vector_unnormalized_with_term():
  """Test create_input_vector_unnormalized with term parameter"""
  
  with tempfile.TemporaryDirectory() as temp_dir:
    generics = ["Generic1", "Generic2"]
    categories = ["Category1", "Category2"]
    processing = {"decay": 0.8, "min_term": 111, "n_terms": 9}
    
    with open(os.path.join(temp_dir, "generics.json"), "w") as f:
      json.dump(generics, f)
    with open(os.path.join(temp_dir, "categories.json"), "w") as f:
      json.dump(categories, f)
    with open(os.path.join(temp_dir, "processing.json"), "w") as f:
      json.dump(processing, f)
    
    creator = CreateInputsOutputs(temp_dir)
    result = creator.create_input_vector_unnormalized(term=113, min_term=111, n_terms=9)
    
    # Check that the correct term position is set
    term_start_idx = 2 * len(generics) + 2 * len(categories)
    assert result[term_start_idx + 2] == 1  # term 113 - 111 = index 2


def test_create_input_vector_unnormalized_with_prev_input_vector():
  """Test create_input_vector_unnormalized with previous input vector for decay"""
  
  with tempfile.TemporaryDirectory() as temp_dir:
    generics = ["Generic1", "Generic2"]
    categories = ["Category1", "Category2"]
    processing = {"decay": 0.8, "min_term": 111, "n_terms": 9}
    
    with open(os.path.join(temp_dir, "generics.json"), "w") as f:
      json.dump(generics, f)
    with open(os.path.join(temp_dir, "categories.json"), "w") as f:
      json.dump(categories, f)
    with open(os.path.join(temp_dir, "processing.json"), "w") as f:
      json.dump(processing, f)
    
    creator = CreateInputsOutputs(temp_dir)
    
    # Create a previous input vector with some values in recent positions
    prev_vector = np.zeros(2 * len(generics) + 2 * len(categories) + 2 + 9)
    prev_vector[2] = 0.5  # previous recent generic value
    prev_vector[6] = 0.3  # previous recent category value
    
    result = creator.create_input_vector_unnormalized(
      prev_input_vector=prev_vector,
      prev_generic="Generic1",
      prev_categories=["Category1"]
    )
    
    # Check decay calculation: new_value = current*(1-decay) + decay*previous
    expected_recent_generic = 1 * 0.2 + 0.8 * 0.5  # 0.2 + 0.4 = 0.6
    expected_recent_category = 1 * 0.2 + 0.8 * 0.3  # 0.2 + 0.24 = 0.44
    
    assert abs(result[2] - expected_recent_generic) < 1e-10
    assert abs(result[6] - expected_recent_category) < 1e-10


def test_create_input_vector_unnormalized_invalid_inputs():
  """Test create_input_vector_unnormalized with invalid inputs"""
  
  with tempfile.TemporaryDirectory() as temp_dir:
    generics = ["Generic1", "Generic2"]
    categories = ["Category1", "Category2"]
    processing = {"decay": 0.8, "min_term": 111, "n_terms": 9}
    
    with open(os.path.join(temp_dir, "generics.json"), "w") as f:
      json.dump(generics, f)
    with open(os.path.join(temp_dir, "categories.json"), "w") as f:
      json.dump(categories, f)
    with open(os.path.join(temp_dir, "processing.json"), "w") as f:
      json.dump(processing, f)
    
    creator = CreateInputsOutputs(temp_dir)
    
    # Test with invalid generic
    result = creator.create_input_vector_unnormalized(prev_generic="InvalidGeneric")
    assert np.all(result[:4] == 0)  # No generic or recent generic should be set
    
    # Test with invalid category
    result = creator.create_input_vector_unnormalized(prev_categories=["InvalidCategory"])
    assert np.all(result[4:8] == 0)  # No category or recent category should be set
    
    # Test with invalid chamber
    result = creator.create_input_vector_unnormalized(chamber="InvalidChamber")
    assert np.all(result[-2:] == 0)  # No chamber should be set