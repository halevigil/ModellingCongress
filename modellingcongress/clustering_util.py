from collections import defaultdict


def cluster(actions,similar,special_cluster_f = lambda x:None,cluster_names_f=lambda x:x):
  special_clusters=defaultdict(list)
  cluster_list=[]
  
  for i,action in enumerate(actions):
    clustered=False
    if action_special_cluster:=special_cluster_f(action):
      special_clusters[action_special_cluster].append(action)
      clustered=True
    else:
      for cluster in reversed(cluster_list):
        if similar(cluster[0],action):
          cluster.append(action)
          clustered=True
          break
    if i%1000==0:
      print(i)
    if not clustered:
      print(action)
      cluster_list.append([action])
  clusters = {cluster_names_f(cluster[0]):cluster for cluster in cluster_list}
  clusters.update(special_clusters)
  return clusters
