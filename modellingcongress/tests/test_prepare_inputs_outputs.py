# # from prepare_inputs_outputs import create_vectors_bill,create_vectors_bill_normalized,concat,normalize
# import pandas as pd
# import numpy as np
# def test_concat():
#   # Test basic list concatenation
#   test_lists = [[1,2], [3,4], [5,6]]
#   result = concat(test_lists)
#   assert result == [1,2,3,4,5,6]
  
#   # Test empty list
#   assert concat([]) == []
  
#   # Test single list
#   assert concat([[1,2]]) == [1,2]

# def test_normalize():
#   # Create test data
#   test_data = pd.DataFrame({
#     'cum_prev_generics': [[1,2,3], [4,5,6]], 
#     'cum_prev_categories': [[1,2], [3,4]]
#   })
  
#   to_normalize = pd.Series({
#     'cum_prev_generics': [2,3,4],
#     'cum_prev_categories': [2,3]
#   })
  
#   # Run normalization
#   result = normalize(to_normalize, test_data)
  
#   # Verify normalization occurred
#   assert isinstance(result, pd.Series)
#   assert 'cum_prev_generics' in result
#   assert 'cum_prev_categories' in result
#   assert len(result['cum_prev_generics']) == 3
#   assert len(result['cum_prev_categories']) == 2

# def test_create_vectors_bill_normalized():
#   # Test data
#   test_df = pd.DataFrame({
#     'generic': ['Action1'],
#     'categories': [['Cat1']],
#     'chamber': ['House'],
#     'term': [111]
#   })
  
#   common_generics = ['Action1']
#   common_categories = ['Cat1']
  
#   # Run function
#   result = create_vectors_bill_normalized(test_df, common_generics, common_categories)
  
#   # Verify output is normalized
#   assert isinstance(result, pd.DataFrame)
#   assert 'cum_prev_generics' in result.columns
#   assert 'cum_prev_categories' in result.columns
# def test_create_vectors_bill_edge_cases():
#   # Test empty dataframe
#   empty_df = pd.DataFrame(columns=['generic','categories','chamber','term'])
#   common_generics = ['Action1']
#   common_categories = ['Cat1'] 
#   result = create_vectors_bill(empty_df, common_generics, common_categories)
#   assert len(result) == 0

#   # Test with unknown generic and category
#   test_df = pd.DataFrame({
#     'generic': ['UnknownAction'],
#     'categories': [['UnknownCat']],
#     'chamber': ['House'],
#     'term': [111]
#   })
#   result = create_vectors_bill(test_df, common_generics, common_categories)
#   assert result.loc[0,'output_generic'][-1] == 1  # Unknown generic mapped to last position
#   assert all(result.loc[0,'output_categories'] == 0)  # Unknown category mapped to zeros
  
#   # Test with null values
#   test_df = pd.DataFrame({
#     'generic': ['Action1'],
#     'categories': [None],
#     'chamber': [None], 
#     'term': [None]
#   })
#   result = create_vectors_bill(test_df, common_generics, common_categories)
#   assert all(result.loc[0,'chamber'] == 0)  # Null chamber maps to zeros
#   assert all(result.loc[0,'term'] == 0)  # Null term maps to zeros
  
#   # Test multiple bills with same generic/category
#   test_df = pd.DataFrame({
#     'generic': ['Action1','Action1'],
#     'categories': [['Cat1'],['Cat1']],
#     'chamber': ['House','House'],
#     'term': [111,111]
#   })
#   result = create_vectors_bill(test_df, common_generics, common_categories)
#   # Verify that cumulative counts increase
#   assert result.loc[1,'cum_prev_generics'][0] > result.loc[0,'cum_prev_generics'][0]
#   assert result.loc[1,'cum_prev_categories'][0] > result.loc[0,'cum_prev_categories'][0]
# def test_create_vectors_bill_edge_cases():
#     # Test empty dataframe
#     empty_df = pd.DataFrame(columns=['generic','categories','chamber','term'])
#     common_generics = ['Action1']
#     common_categories = ['Cat1'] 
#     result = create_vectors_bill(empty_df, common_generics, common_categories)
#     assert len(result) == 0

#     # Test with unknown generic and category
#     test_df = pd.DataFrame({
#       'generic': ['UnknownAction'],
#       'categories': [['UnknownCat']],
#       'chamber': ['House'],
#       'term': [111]
#     })
#     result = create_vectors_bill(test_df, common_generics, common_categories)
#     assert result.loc[0,'output_generic'][-1] == 1  # Unknown generic mapped to last position
#     assert all(result.loc[0,'output_categories'] == 0)  # Unknown category mapped to zeros
    
#     # Test with null values
#     test_df = pd.DataFrame({
#       'generic': ['Action1'],
#       'categories': [None],
#       'chamber': [None], 
#       'term': [None]
#     })
#     result = create_vectors_bill(test_df, common_generics, common_categories)
#     assert all(result.loc[0,'chamber'] == 0)  # Null chamber maps to zeros
#     assert all(result.loc[0,'term'] == 0)  # Null term maps to zeros
    
#     # Test decay factor
#     test_df = pd.DataFrame({
#       'generic': ['Action1','Action1'],
#       'categories': [['Cat1'],['Cat1']],
#       'chamber': ['House','House'],
#       'term': [111,111]
#     })
#     result = create_vectors_bill(test_df, common_generics, common_categories)
#     assert result.loc[1,'recent_generics'][0] > result.loc[0,'recent_generics'][0]
# def test_create_vectors_bill_math_verification():
#   # Test mathematical correctness of exponential decay and cumulative sums
#   test_df = pd.DataFrame({
#     'generic': ['Action1', 'Action2', 'Action1', 'Action2'],
#     'categories': [['Cat1'], ['Cat2'], ['Cat1', 'Cat2'], ['Cat1']],
#     'chamber': ['House', 'Senate', 'House', 'Senate'],
#     'term': [111, 112, 113, 114]
#   })
  
#   common_generics = ['Action1', 'Action2']
#   common_categories = ['Cat1', 'Cat2']
#   alpha = 0.5
  
#   result = create_vectors_bill(test_df, common_generics, common_categories, alpha)
  
#   # Verify first row (no previous data)
#   assert all(result.iloc[0]['recent_generics'] == 0)
#   assert all(result.iloc[0]['cum_prev_generics'] == 0)
#   assert all(result.iloc[0]['recent_categories'] == 0)
#   assert all(result.iloc[0]['cum_prev_categories'] == 0)
  
#   # Verify second row exponential decay calculation
#   # recent_generics = alpha * 0 + alpha * [1,0] = [0.5, 0]
#   expected_recent_generics = np.array([0.5, 0])
#   np.testing.assert_array_equal(result.iloc[1]['recent_generics'], expected_recent_generics)
  
#   # Verify cumulative sum
#   expected_cum_prev = np.array([1, 0])
#   np.testing.assert_array_equal(result.iloc[1]['cum_prev_generics'], expected_cum_prev)
  
#   # Verify third row calculations
#   # Previous was Action2: [0,1], recent becomes alpha*[0.5,0] + alpha*[0,1] = [0.25, 0.5]
#   expected_recent_generics_3 = np.array([0.25, 0.5])
#   np.testing.assert_array_equal(result.iloc[2]['recent_generics'], expected_recent_generics_3)
  
#   # Cumulative: [1,0] + [0,1] = [1,1]
#   expected_cum_prev_3 = np.array([1, 1])
#   np.testing.assert_array_equal(result.iloc[2]['cum_prev_generics'], expected_cum_prev_3)

# def test_create_vectors_bill_category_math():
#   # Test category vector calculations with different alpha
#   test_df = pd.DataFrame({
#     'generic': ['Action1', 'Action1', 'Action1'],
#     'categories': [['Cat1'], ['Cat2'], ['Cat1', 'Cat2']],
#     'chamber': ['House', 'House', 'House'],
#     'term': [111, 111, 111]
#   })
  
#   common_generics = ['Action1']
#   common_categories = ['Cat1', 'Cat2']
#   alpha = 0.8
  
#   result = create_vectors_bill(test_df, common_generics, common_categories, alpha)
  
#   # First row: no previous categories
#   assert all(result.iloc[0]['recent_categories'] == 0)
#   assert all(result.iloc[0]['cum_prev_categories'] == 0)
  
#   # Second row: recent = 0.8*[0,0] + 0.2*[1,0] = [0.2, 0]
#   expected_recent_cat_2 = np.array([0.2, 0])
#   np.testing.assert_array_equal(result.iloc[1]['recent_categories'], expected_recent_cat_2)
  
#   # Third row: recent = 0.8*[0.2,0] + 0.2*[0,1] = [0.16, 0.2]
#   expected_recent_cat_3 = np.array([0.16, 0.2])
#   np.testing.assert_array_equal(result.iloc[2]['recent_categories'], expected_recent_cat_3)

# def test_create_vectors_bill_output_vectors():
#   # Test output vector generation
#   test_df = pd.DataFrame({
#     'generic': ['Action1', 'UnknownAction', 'Action2'],
#     'categories': [['Cat1'], ['UnknownCat'], ['Cat1', 'Cat2']],
#     'chamber': ['House', 'Senate', None],
#     'term': [111, 999, 112]  # 999 is outside MIN_TERM + N_TERMS range
#   })
  
#   common_generics = ['Action1', 'Action2']
#   common_categories = ['Cat1', 'Cat2']
#   alpha = 0.5
  
#   result = create_vectors_bill(test_df, common_generics, common_categories, alpha)
  
#   # First row: Action1 should be [1,0,0] (known generic)
#   expected_output_generic_1 = np.array([1, 0, 0])
#   np.testing.assert_array_equal(result.iloc[0]['output_generic'], expected_output_generic_1)
  
#   # Second row: UnknownAction should be [0,0,1] (unknown generic)
#   expected_output_generic_2 = np.array([0, 0, 1])
#   np.testing.assert_array_equal(result.iloc[1]['output_generic'], expected_output_generic_2)
  
#   # Test chamber encoding
#   expected_chamber_1 = np.array([1, 0])  # House
#   np.testing.assert_array_equal(result.iloc[0]['chamber'], expected_chamber_1)
  
#   expected_chamber_2 = np.array([0, 1])  # Senate
#   np.testing.assert_array_equal(result.iloc[1]['chamber'], expected_chamber_2)
  
#   expected_chamber_3 = np.array([0, 0])  # None
#   np.testing.assert_array_equal(result.iloc[2]['chamber'], expected_chamber_3)
  
#   # Test term encoding
#   expected_term_1 = np.zeros(9)
#   expected_term_1[0] = 1  # term 111 - MIN_TERM(111) = 0
#   np.testing.assert_array_equal(result.iloc[0]['term'], expected_term_1)
  
#   expected_term_3 = np.zeros(9)
#   expected_term_3[1] = 1  # term 112 - MIN_TERM(111) = 1
#   np.testing.assert_array_equal(result.iloc[2]['term'], expected_term_3)

# def test_create_vectors_bill_alpha_extremes():
#   # Test with alpha = 0 (no decay) and alpha = 1 (full decay)
#   test_df = pd.DataFrame({
#     'generic': ['Action1', 'Action1', 'Action1'],
#     'categories': [['Cat1'], ['Cat1'], ['Cat1']],
#     'chamber': ['House', 'House', 'House'],
#     'term': [111, 111, 111]
#   })
  
#   common_generics = ['Action1']
#   common_categories = ['Cat1']
  
#   # Test alpha = 0 (no exponential weighting)
#   result_no_decay = create_vectors_bill(test_df, common_generics, common_categories, 0.0)
#   # With alpha=0, recent should stay at 0 since alpha*recent + alpha*prev = 0
#   assert all(result_no_decay.iloc[1]['recent_generics'] == 0)
#   assert all(result_no_decay.iloc[2]['recent_generics'] == 0)
  
#   # Test alpha = 1 (full previous value)
#   result_full_decay = create_vectors_bill(test_df, common_generics, common_categories, 1.0)
#   # With alpha=1: recent = 1*recent + 1*prev
#   # Row 1: recent = 1*[0] + 1*[1] = [1]
#   # Row 2: recent = 1*[1] + 1*[1] = [2]
#   expected_recent_full = np.array([1.0])
#   np.testing.assert_array_equal(result_full_decay.iloc[1]['recent_generics'], expected_recent_full)
#   expected_recent_full_2 = np.array([2.0])
#   np.testing.assert_array_equal(result_full_decay.iloc[2]['recent_generics'], expected_recent_full_2)

# def test_create_vectors_bill_multiple_categories():
#   # Test handling of multiple categories in single action
#   test_df = pd.DataFrame({
#     'generic': ['Action1'],
#     'categories': [['Cat1', 'Cat2', 'Cat3']],
#     'chamber': ['House'],
#     'term': [111]
#   })
  
#   common_generics = ['Action1']
#   common_categories = ['Cat1', 'Cat2', 'Cat3', 'Cat4']
#   alpha = 0.5
  
#   result = create_vectors_bill(test_df, common_generics, common_categories, alpha)
  
#   # Should have 1s for Cat1, Cat2, Cat3 and 0 for Cat4
#   expected_output_categories = np.array([1, 1, 1, 0])
#   np.testing.assert_array_equal(result.iloc[0]['output_categories'], expected_output_categories)

# def test_create_vectors_bill_dataframe_structure():
#   # Test that output DataFrame has correct structure
#   test_df = pd.DataFrame({
#     'generic': ['Action1'],
#     'categories': [['Cat1']],
#     'chamber': ['House'],
#     'term': [111]
#   })
  
#   common_generics = ['Action1', 'Action2']
#   common_categories = ['Cat1', 'Cat2']
#   alpha = 0.5
  
#   result = create_vectors_bill(test_df, common_generics, common_categories, alpha)
  
#   # Check DataFrame structure
#   expected_columns = ['recent_generics', 'cum_prev_generics', 'recent_categories', 
#               'cum_prev_categories', 'term', 'chamber', 'output_generic', 'output_categories']
#   assert all(col in result.columns for col in expected_columns)
#   assert len(result) == len(test_df)
  
#   # Check vector dimensions
#   assert len(result.iloc[0]['recent_generics']) == len(common_generics)
#   assert len(result.iloc[0]['cum_prev_generics']) == len(common_generics)
#   assert len(result.iloc[0]['recent_categories']) == len(common_categories)
#   assert len(result.iloc[0]['cum_prev_categories']) == len(common_categories)
#   assert len(result.iloc[0]['output_generic']) == len(common_generics) + 1  # +1 for unknown
#   assert len(result.iloc[0]['output_categories']) == len(common_categories)
#   assert len(result.iloc[0]['term']) == 9  # N_TERMS
#   assert len(result.iloc[0]['chamber']) == 2
