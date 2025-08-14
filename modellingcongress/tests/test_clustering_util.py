import pytest
import sys
# from modellingcongress.clustering_util import cluster
from ..clustering_util import cluster

def test_basic_clustering():
  actions = ['a', 'b', 'a', 'c', 'b']
  def similar(x, y): return x == y
  result = cluster(actions, similar)
  assert len(result) == 3
  assert sorted(result.keys()) == ['a', 'b', 'c']
  assert result['a'] == ['a', 'a']
  assert result['b'] == ['b', 'b']
  assert result['c'] == ['c']

def test_empty_list():
  assert cluster([], lambda x,y: x==y) == {}

def test_special_cluster():
  actions = ['a1', 'a2', 'b1', 'special']
  def special_f(x): return 'special_cluster' if x == 'special' else None
  def similar(x, y): return x[0] == y[0]
  result = cluster(actions, similar, special_f)
  assert len(result) == 3
  assert 'special_cluster' in result
  assert result['special_cluster'] == ['special']
  assert sorted(len(cluster) for cluster in result.values()) == [1, 1, 2]

def test_custom_cluster_names():
  actions = ['test1', 'test2', 'test1']
  def similar(x, y): return x == y
  def name_f(x): return f"cluster_{x}"
  result = cluster(actions, similar, cluster_names_f=name_f)
  assert list(result.keys()) == ['cluster_test1', 'cluster_test2']
  assert result['cluster_test1'] == ['test1', 'test1']

def test_all_similar():
  actions = ['x', 'x', 'x', 'x']
  def similar(x, y): return True
  result = cluster(actions, similar)
  assert len(result) == 1
  assert list(result.values())[0] == actions
if __name__=="__main__":
  test_basic_clustering()
  test_empty_list()
  test_custom_cluster_names()
  test_special_cluster()
  test_all_similar()