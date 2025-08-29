# File for inspection of model weights
import json
import pandas as pd
import numpy as np

import sklearn.preprocessing
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import sklearn
from collections import defaultdict, Counter
from IPython.display import display



with open("outputs/preprocess5/inference/generics.json","r") as file:
  generics = json.load(file)
with open("outputs/preprocess5/inference/categories.json","r") as file:
  categories = json.load(file)


n_terms=9
min_term=111
from collections import defaultdict
import torch
import itertools
weights = torch.load("outputs/preprocess5/models/lr3e-04_lassoweight1e-06_batch256/epoch10.pt")["model"]["weight"]
weights = weights.detach().numpy() 
print(weights.shape)
last_chains=[]
recent_chains=[]
cumulative_chains=[]
recent_plus_s_chains=[]
recent_s_diff_chains=[]
term_chains=[]
for i,name1 in enumerate(generics):
  for j,name2 in enumerate(generics):
    if name1 in ["Referred to [SUBCOMMITTEE]."]:

    
      last_weight = float(weights[j][i])
      recent_weight = float(weights[j][i+len(generics)])
      # cum_weight = float(weights[j][i+len(generics)])
      last_chains.append((last_weight,name1,name2))
      recent_chains.append((recent_weight,name1,name2))
    # cumulative_chains.append((cum_weight,name1,name2))
# for i,name1 in enumerate(categories):
#   for j,name2 in enumerate(generics):
    
#     start=2*len(generics)
#     last_weight = float(weights[j][start+i])
#     recent_weight = float(weights[j][start+len(categories)+i])
#     # cum_weight = float(weights[j][start+2*len(categories)+i])
#     last_chains.append((last_weight,name1,name2))
#     recent_chains.append((recent_weight,name1,name2))
    # cumulative_chains.append((cum_weight,name1,name2))
for term in range(n_terms):
  for j,generic2 in enumerate(generics):
    term_chains.append((float(weights[j][term+2*len(generics)+2*len(categories)]),str(term*2+2009)+"-"+str(term*2+2010),generic2))
chamber_chains=[]
for j,generic in enumerate(generics):
    chamber_chains.append((float(weights[j][2*len(generics)+2*len(categories)+n_terms]),"House",generic))
for j,generic in enumerate(generics):
    chamber_chains.append((float(weights[j][1+2*len(generics)+2*len(categories)+n_terms]),"Senate",generic))
all_chains = recent_chains+cumulative_chains+term_chains+chamber_chains

def display_chains(chains,name=""):
  remove_negative_selfchains=[x for x in chains if x[1]!=x[2] or x[0]>0]
  top100=sorted(remove_negative_selfchains,key=lambda x:abs(x[0]),reverse=True)[:100]
  if name!="":
    print(name)
  display([str(param[0])+" "+param[1]+" -> "+param[2] for param in top100])
  display(sum(x[0] for x in sorted(chains,key=lambda x: abs(x[0]),reverse=True)[-10:])/10)
if __name__=="__main__":
  # display_chains(recent_plus_s_chains,"recent+cum:")
  display_chains(last_chains,"last:")
  display_chains(recent_chains,"recent:")
  display_chains(cumulative_chains,"cumulative:")

  display_chains(term_chains,"terms:")
  display_chains(chamber_chains,"chamber:")

  