import json
import pandas as pd
import numpy as np

import sklearn.preprocessing
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import sklearn
from collections import defaultdict, Counter



#   for line in file:
#     if line.strip():  # Skip empty lines
#       response = json.loads(line)
#       llm_generics.append(response["response"]["body"]["output"][0]["content"][0]["text"])
      
# history_df["llm_generic"] = history_df["generic"].map(dict(zip(generic_names, llm_generics)))
# display(generic_map)
with open("/Users/gilhalevi/Library/CloudStorage/OneDrive-Personal/Code/ModellingCongress/outputs/preprocess0/inference/generics.json","r") as file:
  generics = json.load(file)
with open("/Users/gilhalevi/Library/CloudStorage/OneDrive-Personal/Code/ModellingCongress/outputs/preprocess0/inference/categories.json","r") as file:
  categories = json.load(file)


n_terms=9
min_term=111
from collections import defaultdict
import torch
import itertools
# model.load_state_dict(torch.load("./../outputs/models/08-04_lr1e-5_beta.01/epoch99.pt")["model"])
weights = torch.load("/Users/gilhalevi/Library/CloudStorage/OneDrive-Personal/Code/ModellingCongress/outputs/preprocess0/models/lr3e-04_lassoweight0e+00_batch256/epoch115.pt")["model"]["weight"]
weights = weights.detach().numpy() 
print(weights.shape)
recent_chains=[]
cumulative_chains=[]
recent_plus_s_chains=[]
recent_s_diff_chains=[]
term_chains=[]
for i,name1 in itertools.chain(enumerate(generics),enumerate(categories,2*len(generics))):
  for j,name2 in enumerate(generics+categories):
    # print(i)
    pred_weight = float(weights[j][i])
    preds_weight = float(weights[j][i+len(generics) if i<len(generics) else i+len(categories)])
    recent_chains.append((pred_weight,name1,name2))
    cumulative_chains.append((preds_weight,name1,name2))
    # recent_s_diff_chains.append((abs(pred_weight-preds_weight)*(pred_weight+preds_weight),generic1,generic2))
    # recent_plus_s_chains.append(((pred_weight+preds_weight),generic1,generic2))
for term in range(n_terms):
  for j,generic2 in enumerate(generics+categories):
    term_chains.append((float(weights[j][term+2*len(generics)+2*len(categories)]),str(term*2+2009)+"-"+str(term*2+2010),generic2))
chamber_chains=[]
for j,generic in enumerate(generics+categories):
    chamber_chains.append((float(weights[j][2*len(generics)+2*len(categories)+n_terms]),"House",generic))
for j,generic in enumerate(generics+categories):
    chamber_chains.append((float(weights[j][1+2*len(generics)+2*len(categories)+n_terms]),"Senate",generic))
all_chains = recent_chains+cumulative_chains+term_chains+chamber_chains

def display_chains(chains,name=""):
  remove_negative_selfchains=[x for x in chains if x[1]!=x[2] or x[0]>0]
  top100=sorted(remove_negative_selfchains,key=lambda x:x[0],reverse=True)[:100]
  if name!="":
    print(name)
  display([str(param[0])+" "+param[1]+" -> "+param[2] for param in top100])
  display(sum(x[0] for x in sorted(chains,key=lambda x: abs(x[0]),reverse=True)[-10:])/10)
if __name__=="__main__":
  # display_chains(recent_plus_s_chains,"recent+cum:")
  display_chains(recent_chains,"recent:")
  display_chains(cumulative_chains,"cumulative:")

  display_chains(term_chains,"terms:")
  display_chains(chamber_chains,"chamber:")

  