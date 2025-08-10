import json
import pandas as pd
import numpy as np

import sklearn.preprocessing
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import sklearn
from collections import defaultdict, Counter
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
# with open("./outputs/bucket_names_output.jsonl", "r") as file:
#   for line in file:
#     if line.strip():  # Skip empty lines
#       response = json.loads(line)
#       llm_buckets.append(response["response"]["body"]["output"][0]["content"][0]["text"])
      
# history_df["llm_bucket"] = history_df["bucket"].map(dict(zip(bucket_names, llm_buckets)))
# display(bucket_map)
unbucketed=[]
for action in history_df["action"]:
  if action not in bucket_map:
    unbucketed.append(action)
# print(len(unbucketed))
# Add next bill_id column to compare
history_df["bucket"]=history_df["action"].apply(lambda action:bucket_map[action])
history_df["extra_buckets"]=history_df["action"].apply(lambda action:extra_bucket_map[action])
bills={bill_id:group for (bill_id,group) in history_df.groupby("bill_id")}
bill_ids = list(bills.keys())

MIN_TERM=min(history_df["term"])
MAX_TERM=max(history_df["term"])
N_TERMS=MAX_TERM-MIN_TERM+1


class ActionDataset(Dataset):
  def __init__(self,path):
    data = np.load(path,allow_pickle=True)["arr_0"]
    self.inputs = [np.concatenate([entry["predecessor_buckets"],entry["predecessors_buckets"],entry["predecessor_extra_buckets"],entry["predecessors_extra_buckets"],entry["term"],entry["chamber"]]) for entry in data]
    self.inputs = sklearn.preprocessing.StandardScaler().fit_transform(self.inputs)
    self.outputs = [np.concatenate((entry["output_bucket"],entry["output_extra_buckets"])) for entry in data]
  def __len__(self):
    return len(self.inputs)
  def __getitem__(self,idx):
    return self.inputs[idx],self.outputs[idx]

ds = ActionDataset("outputs/prediction_vecs_08-04_withextras.npz")
# print(len(ds))
train_dataset,test_dataset,val_dataset = torch.utils.data.random_split(ds,[0.6,0.2,0.2])


from collections import defaultdict
import torch
import itertools
# model.load_state_dict(torch.load("outputs/models/08-04_lr1e-5_beta.01/epoch99.pt")["model"])
model = torch.nn.Linear(len(common_bucket_names)*2+len(common_extra_bucket_names)*2+N_TERMS+2,len(common_bucket_names)+1+len(common_extra_bucket_names))
model.load_state_dict(torch.load("outputs/models/08-07_lr3e-04_lassoweight1e-05_batch256_extra/epoch160.pt")["model"])

weights = model.weight.detach().numpy() 
predecessor_chains=[]
predecessors_chains=[]
predecessor_plus_s_chains=[]
predecessor_s_diff_chains=[]
term_chains=[]
common_extra_bucket_names_prefixed=["Extra Bucket: "+x for x in common_extra_bucket_names]
output_buckets = common_bucket_names+["No Bucket"]+common_extra_bucket_names_prefixed
for i,bucket1 in itertools.chain(enumerate(common_bucket_names),enumerate(common_extra_bucket_names_prefixed,2*len(common_bucket_names))):
  for j,bucket2 in enumerate(output_buckets):
    # print(i)
    pred_weight = float(weights[j][i])
    preds_weight = float(weights[j][i+len(common_bucket_names) if i<len(common_bucket_names) else i+len(common_extra_bucket_names)])
    predecessor_chains.append((pred_weight,bucket1,bucket2))
    predecessors_chains.append((preds_weight,bucket1,bucket2))
    predecessor_s_diff_chains.append((abs(pred_weight-preds_weight)*(pred_weight+preds_weight),bucket1,bucket2))
    predecessor_plus_s_chains.append(((pred_weight+preds_weight),bucket1,bucket2))
for term in range(N_TERMS):
  for j,bucket2 in enumerate(output_buckets):
    term_chains.append((float(weights[j][term+2*len(common_bucket_names)+2*len(common_extra_bucket_names)]),str(term*2+2009)+"-"+str(term*2+2010),bucket2))
chamber_chains=[]
for j,bucket in enumerate(output_buckets):
    chamber_chains.append((float(weights[j][1+term+2*len(common_bucket_names)+2*len(common_extra_bucket_names)]),"House",bucket))
for j,bucket in enumerate(output_buckets):
    chamber_chains.append((float(weights[j][2+term+2*len(common_bucket_names)+2*len(common_extra_bucket_names)]),"Senate",bucket))
all_chains = predecessor_chains+predecessors_chains+term_chains+chamber_chains

def display_chains(chains,name=""):
  remove_negative_selfchains=[x for x in chains if x[1]!=x[2] or x[0]>0]
  top100=sorted(remove_negative_selfchains,key=lambda x:x[0],reverse=True)[:100]
  if name!="":
    print(name)
  display([str(param[0])+" "+param[1]+" -> "+param[2] for param in top100])
  display(sum(x[0] for x in sorted(chains,key=lambda x: abs(x[0]),reverse=True)[-10:])/10)
if __name__=="__main__":
  display_chains(predecessor_s_diff_chains,"all chains:")
  display_chains(predecessor_s_diff_chains,"diffs:")
  display_chains(predecessor_plus_s_chains,"predecessor+predecessors:")
  display_chains(predecessor_chains,"predecessor:")
  display_chains(predecessors_chains,"predecessors:")

  display_chains(term_chains,"terms:")
  display_chains(chamber_chains,"chamber:")

  