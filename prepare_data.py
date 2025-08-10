import json
import pandas as pd
from collections import defaultdict, Counter
import sklearn
import sklearn.model_selection
import random

with open("outputs/buckets_08-04_manual-llm-manual.json","r") as file:
  buckets = json.load(file)
with open("outputs/extra_buckets_08-04.json","r") as file:
  extra_buckets = json.load(file)


datasets = [f"data/US/{term*2+2009-222}-{term*2+2010-222}_{term}th_Congress/csv/history.csv" for term in range(111,120)]
history_df_by_term=[pd.read_csv(ds) for ds in datasets]
for i,df in enumerate(history_df_by_term):
  df["term"]=i+111
history_df=pd.concat(history_df_by_term)


bucket_map={}
for name in buckets:
  for action in buckets[name]:
    bucket_map[action]=name
extra_bucket_map=defaultdict(list)
n_read_twice=0
for name in extra_buckets:
  for i,action in enumerate(extra_buckets[name]):
    extra_bucket_map[action].append(name)
bucket_lens=Counter()
for action in history_df["action"]:
  bucket_lens[bucket_map[action]]+=1
extra_bucket_lens=Counter()
for i,action in enumerate(history_df["action"]):
  for extra_bucket in extra_bucket_map[action]:
    extra_bucket_lens[extra_bucket]+=1
common_bucket_names = [name for name in buckets.keys() if bucket_lens[name]>=50]
common_extra_bucket_names = [name for name in extra_buckets.keys() if extra_bucket_lens[name]>=50]
common_bucket_names_inv={name:i for i,name in enumerate(common_bucket_names)}
common_extra_bucket_names_inv={name:i for i,name in enumerate(common_extra_bucket_names)}


history_df["bucket"]=history_df["action"].apply(lambda action:bucket_map[action])
history_df["extra_buckets"]=history_df["action"].apply(lambda action:extra_bucket_map[action])
history_df.to_csv("outputs/data/all_data.csv",index=False)
bills={bill_id:group for (bill_id,group) in history_df.groupby("bill_id")}
bill_ids = list(bills.keys())


random.seed(2430)
random.shuffle(bill_ids)
train_bill_ids = bill_ids[:int(0.8*len(bill_ids))]
test_bill_ids = bill_ids[int(0.8*len(bill_ids)):]
train_bills = [bills[bill_id] for bill_id in train_bill_ids]
train_dataset = pd.concat(train_bills)
test_bills = [bills[bill_id] for bill_id in test_bill_ids]
test_dataset = pd.concat(train_bills)

train_dataset.to_csv("outputs/data/train_data.csv",index=False)
test_dataset.to_csv("outputs/data/test_data.csv",index=False) 
with open("outputs/common_bucket_names.json","w") as file:
  json.dump(common_bucket_names,file)

with open("outputs/common_extra_bucket_names.json","w") as file:
  json.dump(common_extra_bucket_names,file)