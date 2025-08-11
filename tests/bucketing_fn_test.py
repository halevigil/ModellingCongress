import pytest
from ..bucketing_fn import bucket

def test_basic_bucketing():
  actions = ['a', 'b', 'a', 'c', 'b']
  def similar(x, y): return x == y
  result = bucket(actions, similar)
  assert len(result) == 3
  assert sorted(result.keys()) == ['a', 'b', 'c']
  assert result['a'] == ['a', 'a']
  assert result['b'] == ['b', 'b']
  assert result['c'] == ['c']

def test_empty_list():
  assert bucket([], lambda x,y: x==y) == {}

def test_special_bucket():
  actions = ['a1', 'a2', 'b1', 'special']
  def special_f(x): return 'special_bucket' if x == 'special' else None
  def similar(x, y): return x[0] == y[0]
  result = bucket(actions, similar, special_f)
  assert len(result) == 3
  assert 'special_bucket' in result
  assert result['special_bucket'] == ['special']
  assert sorted(len(bucket) for bucket in result.values()) == [1, 1, 2]

def test_custom_bucket_names():
  actions = ['test1', 'test2', 'test1']
  def similar(x, y): return x == y
  def name_f(x): return f"bucket_{x}"
  result = bucket(actions, similar, bucket_names_f=name_f)
  assert list(result.keys()) == ['bucket_test1', 'bucket_test2']
  assert result['bucket_test1'] == ['test1', 'test1']

def test_all_similar():
  actions = ['x', 'x', 'x', 'x']
  def similar(x, y): return True
  result = bucket(actions, similar)
  assert len(result) == 1
  assert list(result.values())[0] == actions
if __name__=="__main__":
  test_basic_bucketing()
  test_empty_list()
  test_custom_bucket_names()
  test_special_bucket()
  test_all_similar()