import json
import pandas as pd
import numpy as np

import sklearn.preprocessing
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import sklearn
from collections import defaultdict, Counter
with open("../outputs/generics_08-04_manual-llm-manual.json","r") as file:
  generics = json.load(file)
with open("../outputs/categories_08-04.json","r") as file:
  categories = json.load(file)




datasets = [f"data/{term*2+2009-222}-{term*2+2010-222}_{term}th_Congress/csv/history.csv" for term in range(111,120)]
history_df_by_term=[pd.read_csv(ds) for ds in datasets]
for i,df in enumerate(history_df_by_term):
  df["term"]=i+111
history_df=pd.concat(history_df_by_term)


generic_map={}
for name in generics:
  for action in generics[name]:
    generic_map[action]=name
category_map=defaultdict(list)
n_read_twice=0
for name in categories:
  for i,action in enumerate(categories[name]):
    category_map[action].append(name)
generic_lens=Counter()
for action in history_df["action"]:
  generic_lens[generic_map[action]]+=1
category_lens=Counter()
for i,action in enumerate(history_df["action"]):
  for category in category_map[action]:
    category_lens[category]+=1
common_generic_names = [name for name in generics.keys() if generic_lens[name]>=50]
common_category_names = [name for name in categories.keys() if category_lens[name]>=50]
common_generic_names_inv={name:i for i,name in enumerate(common_generic_names)}
common_category_names_inv={name:i for i,name in enumerate(common_category_names)}
# with open("./../outputs/generic_names_output.jsonl", "r") as file:
#   for line in file:
#     if line.strip():  # Skip empty lines
#       response = json.loads(line)
#       llm_generics.append(response["response"]["body"]["output"][0]["content"][0]["text"])
      
# history_df["llm_generic"] = history_df["generic"].map(dict(zip(generic_names, llm_generics)))
# display(generic_map)
ungenericed=[]
for action in history_df["action"]:
  if action not in generic_map:
    ungenericed.append(action)
# print(len(ungenericed))
# Add next bill_id column to compare
history_df["generic"]=history_df["action"].apply(lambda action:generic_map[action])
history_df["categories"]=history_df["action"].apply(lambda action:category_map[action])
bills={bill_id:group for (bill_id,group) in history_df.groupby("bill_id")}
bill_ids = list(bills.keys())

MIN_TERM=min(history_df["term"])
MAX_TERM=max(history_df["term"])
N_TERMS=MAX_TERM-MIN_TERM+1


class ActionDataset(Dataset):
  def __init__(self,path):
    data = np.load(path,allow_pickle=True)["arr_0"]
    self.inputs = [np.concatenate([entry["predecessor_generics"],entry["predecessors_generics"],entry["predecessor_categories"],entry["predecessors_categories"],entry["term"],entry["chamber"]]) for entry in data]
    self.inputs = sklearn.preprocessing.StandardScaler().fit_transform(self.inputs)
    self.../outputs/ = [np.concatenate((entry["output_generic"],entry["output_categories"])) for entry in data]
  def __len__(self):
    return len(self.inputs)
  def __getitem__(self,idx):
    return self.inputs[idx],self.../outputs/[idx]

ds = ActionDataset("../outputs/prediction_vecs_08-04_withextras.npz")
# print(len(ds))
train_dataset,test_dataset,val_dataset = torch.utils.data.random_split(ds,[0.6,0.2,0.2])


from collections import defaultdict
import torch
import itertools
# model.load_state_dict(torch.load("../outputs/models/08-04_lr1e-5_beta.01/epoch99.pt")["model"])
model = torch.nn.Linear(len(common_generic_names)*2+len(common_category_names)*2+N_TERMS+2,len(common_generic_names)+1+len(common_category_names))
model.load_state_dict(torch.load("../outputs/models/08-07_lr3e-04_lassoweight1e-05_batch256_extra/epoch160.pt")["model"])

weights = model.weight.detach().numpy() 
predecessor_chains=[]
predecessors_chains=[]
predecessor_plus_s_chains=[]
predecessor_s_diff_chains=[]
term_chains=[]
common_category_names_prefixed=["Extra generic: "+x for x in common_category_names]
output_generics = common_generic_names+["Miscellaneous"]+common_category_names_prefixed
for i,generic1 in itertools.chain(enumerate(common_generic_names),enumerate(common_category_names_prefixed,2*len(common_generic_names))):
  for j,generic2 in enumerate(output_generics):
    # print(i)
    pred_weight = float(weights[j][i])
    preds_weight = float(weights[j][i+len(common_generic_names) if i<len(common_generic_names) else i+len(common_category_names)])
    predecessor_chains.append((pred_weight,generic1,generic2))
    predecessors_chains.append((preds_weight,generic1,generic2))
    predecessor_s_diff_chains.append((abs(pred_weight-preds_weight)*(pred_weight+preds_weight),generic1,generic2))
    predecessor_plus_s_chains.append(((pred_weight+preds_weight),generic1,generic2))
for term in range(N_TERMS):
  for j,generic2 in enumerate(output_generics):
    term_chains.append((float(weights[j][term+2*len(common_generic_names)+2*len(common_category_names)]),str(term*2+2009)+"-"+str(term*2+2010),generic2))
chamber_chains=[]
for j,generic in enumerate(output_generics):
    chamber_chains.append((float(weights[j][1+term+2*len(common_generic_names)+2*len(common_category_names)]),"House",generic))
for j,generic in enumerate(output_generics):
    chamber_chains.append((float(weights[j][2+term+2*len(common_generic_names)+2*len(common_category_names)]),"Senate",generic))
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

  