from collections import defaultdict


def bucket(actions,similar,special_bucket_f = lambda x:None,bucket_names_f=lambda x:x):
  special_buckets=defaultdict(list)
  bucket_list=[]
  
  for i,action in enumerate(actions):
    bucketed=False
    if action_special_bucket:=special_bucket_f(action):
      special_buckets[action_special_bucket].append(action)
      bucketed=True
    else:
      for bucket in reversed(bucket_list):
        if similar(bucket[0],action):
          bucket.append(action)
          bucketed=True
          break
    if i%1000==0:
      print(i)
    if not bucketed:
      print(action)
      bucket_list.append([action])
  buckets = {bucket_names_f(bucket[0]):bucket for bucket in bucket_list}
  buckets.update(special_buckets)
  return buckets
