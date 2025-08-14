from ..modellingcongress.categorize import categorize, make_categories
def test_categorize_empty():
  assert categorize("") == []

def test_categorize_simple():
  assert "debate" in categorize("GENERAL DEBATE")
  assert r"^GENERAL DEBATE" in categorize("GENERAL DEBATE")

def test_categorize_case_sensitivity():
  assert "debate" in categorize("general debate")
  assert r"^GENERAL DEBATE" in categorize("GENERAL DEBATE")

def test_categorize_multiple_matches():
  action = "Motion to waive all points of order"
  result = categorize(action)
  assert r"^Motion to waive" in result
  assert "points? of order" in result

def test_make_categories_empty():
  assert make_categories([], categorize) == {}

def test_make_categories_simple():
  actions = ["GENERAL DEBATE", "Motion to waive"]
  result = make_categories(actions, categorize)
  assert "debate" in result
  assert r"^GENERAL DEBATE" in result
  assert r"^Motion to waive" in result
  assert result[r"^GENERAL DEBATE"] == ["GENERAL DEBATE"]
  assert result[r"^Motion to waive"] == ["Motion to waive"]

def test_make_categories_multiple_actions():
  actions = [
    "GENERAL DEBATE",
    "Motion to waive",
    "Motion to waive all points of order"
  ]
  result = make_categories(actions, categorize)
  assert len(result[r"^Motion to waive"]) == 2
  assert "GENERAL DEBATE" in result[r"^GENERAL DEBATE"]
  assert "points? of order" in result