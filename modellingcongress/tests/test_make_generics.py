import os
from make_generics import process_action, edit_distance,edit_distance_below,similar_after_processing,special_generics_f



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


def test_special_generics_f():
  assert special_generics_f("H.Amdt.123 amendment (whatever) in the nature of a substitute offered by Someone") == \
       "H.Amdt. in the nature of a substitute offered by"
  
  assert special_generics_f("H.Amdt.456 amendment (text) offered by Someone") == \
       "H.Amdt. offered by"
  
  assert special_generics_f("S.Amdt.789 amendment SA 123 proposed by Someone") == \
       "S.Amdt. offered by"
  
  assert special_generics_f("Some other action") == None
def test_process_action():
  # Mock committee data for testing
  committee_search_item = {
    "House Energy Committee": "Energy",
    "Senate Commerce Committee": "Commerce",
    "House Judiciary Subcommittee": "Judiciary"
  }
  
  committee_spellings = {
    "House Energy Committee": ["House Energy Committee", "Energy Committee", "Committee on Energy"],
    "Senate Commerce Committee": ["Senate Commerce Committee", "Commerce Committee", "Committee on Commerce"],
    "House Judiciary Subcommittee": ["House Judiciary Subcommittee", "Judiciary Subcommittee"]
  }
  
  # Test basic committee replacement
  assert process_action(
    "Referred to House Energy Committee", 
    committee_search_item, 
    committee_spellings
  ) == "referred to committee"
  
  # Test subcommittee replacement
  assert process_action(
    "Referred to House Judiciary Subcommittee", 
    committee_search_item, 
    committee_spellings
  ) == "referred to subcommittee"
  
  # Test number removal
  assert process_action(
    "Action taken on 123 items", 
    committee_search_item, 
    committee_spellings
  ) == "action taken on items"
  
  # Test representative name removal
  assert process_action(
    "Amendment offered by Mr. Smith", 
    committee_search_item, 
    committee_spellings
  ) == "amendment offered by representative"
  
  # Test time replacement
  assert process_action(
    "Debate for 2 hours", 
    committee_search_item, 
    committee_spellings
  ) == "debate for time"
  
  # Test parenthetical removal
  assert process_action(
    "Bill passed (vote 123-45)", 
    committee_search_item, 
    committee_spellings
  ) == "bill passed"
  
  # Test punctuation and multiple spaces cleanup
  assert process_action(
    "Action,  taken-  with.  multiple   spaces", 
    committee_search_item, 
    committee_spellings
  ) == "action taken with multiple spaces"
  
  # Test case insensitive processing
  assert process_action(
    "REFERRED TO COMMITTEE ON ENERGY", 
    committee_search_item, 
    committee_spellings
  ) == "referred to committee"
  
  # Test committees -> committee replacement
  assert process_action(
    "Sent to committees for review", 
    committee_search_item, 
    committee_spellings
  ) == "sent to committee for review"
def test_similar_after_processing():
  # Mock committee data for testing
  committee_search_item = {
    "House Energy Committee": "Energy",
    "Senate Commerce Committee": "Commerce",
    "House Judiciary Subcommittee": "Judiciary"
  }
  
  committee_spellings = {
    "House Energy Committee": ["House Energy Committee", "Energy Committee", "Committee on Energy"],
    "Senate Commerce Committee": ["Senate Commerce Committee", "Commerce Committee", "Committee on Commerce"],
    "House Judiciary Subcommittee": ["House Judiciary Subcommittee", "Judiciary Subcommittee"]
  }
  
  # Create a mock process function
  def mock_process(action):
    return process_action(action, committee_search_item, committee_spellings)
  
  # Test similar actions with small differences
  assert similar_after_processing(
    "Referred to House Energy Committee", 
    "Referred to Energy Committee", 
    mock_process, 
    0.2
  ) == True
  
  # Test actions that differ only in numbers
  assert similar_after_processing(
    "Amendment 123 offered by Mr. Smith", 
    "Amendment 456 offered by Ms. Jones", 
    mock_process, 
    0.2
  ) == True
  
  # Test actions with time differences
  assert similar_after_processing(
    "Debate for 2 hours", 
    "Debate for 3 hours", 
    mock_process, 
    0.2
  ) == True
  
  # Test actions with parenthetical differences
  assert similar_after_processing(
    "Bill passed (vote 123-45)", 
    "Bill passed (unanimous)", 
    mock_process, 
    0.2
  ) == True
  
  # Test completely different actions
  assert similar_after_processing(
    "Bill introduced", 
    "Committee markup", 
    mock_process, 
    0.2
  ) == False
  
  # Test subcommittee vs committee distinction (mustmatch requirement)
  assert similar_after_processing(
    "Referred to House Judiciary Subcommittee", 
    "Referred to House Energy Committee", 
    mock_process, 
    0.2
  ) == False
  
  # Test identical actions
  assert similar_after_processing(
    "Bill passed", 
    "Bill passed", 
    mock_process, 
    0.2
  ) == True
  
  
  assert similar_after_processing(
    "Committee meeting scheduled", 
    "Subcommittee hearing announced", 
    mock_process, 
    0.3
  ) == False
  
  # Test case sensitivity
  assert similar_after_processing(
    "BILL PASSED", 
    "bill passed", 
    mock_process, 
    0.2
  ) == True