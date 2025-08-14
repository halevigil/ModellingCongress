import os
from make_generics import process_action, edit_distance,edit_distance_below,similar_after_processing,special_generics_f


def test_process_action():
  # Test basic committee name removal
  assert process_action("Referred to House Committee on Energy and Commerce") == "referred to committee"
  assert process_action("Referred to Senate Committee on Finance") == "referred to committee"

  # Test subcommittee handling
  assert process_action("Referred to Subcommittee on Energy") == "referred to subcommittee"
  
  # Test number removal
  assert process_action("Passed House 234-193") == "passed house"
  
  # Test representative name removal 
  assert process_action("Motion by Mr. Smith") == "motion by representative"
  assert process_action("Motion by Representative Jones") == "motion by representative"
  
  # Test time removal
  assert process_action("Debate for 2 hours") == "debate for time"
  assert process_action("Debate for 30 minutes") == "debate for time"
  
  # Test parenthetical removal
  assert process_action("Motion to recommit (with instructions)") == "motion to recommit"

def test_edit_distance():
  assert edit_distance("cat", "bat") == 1
  assert edit_distance("kitten", "sitting") == 3
  assert edit_distance("", "abc") == 3
  assert edit_distance("abc", "") == 3
  assert edit_distance("", "") == 0

def test_edit_distance_below():
  assert edit_distance_below("cat", "bat", 1) == True
  assert edit_distance_below("cat", "dog", 1) == False
  assert edit_distance_below("kitten", "sitting", 3) == True
  assert edit_distance_below("kitten", "sitting", 2) == False
  assert edit_distance_below("", "", 0) == True

def test_similar_after_processing():
  # Test similar actions
  assert similar_after_processing(
    "Referred to House Committee on Energy", 
    "Referred to House Committee on Commerce",
    0.3
  ) == True

  # Test dissimilar actions
  assert similar_after_processing(
    "Referred to Subcommittee", 
    "Referred to Committee",
    0.3
  ) == False

def test_special_generics_f():
  assert special_generics_f("H.Amdt.123 amendment (whatever) in the nature of a substitute offered by Someone") == \
       "H.Amdt. in the nature of a substitute offered by"
  
  assert special_generics_f("H.Amdt.456 amendment (text) offered by Someone") == \
       "H.Amdt. offered by"
  
  assert special_generics_f("S.Amdt.789 amendment SA 123 proposed by Someone") == \
       "S.Amdt. offered by"
  
  assert special_generics_f("Some other action") == None