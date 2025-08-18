import pandas as pd
from collections import defaultdict
import random
import json
import os
import argparse
import numpy as np
import sklearn





def create_next_vector_unnormalized(decay,all_generics,all_categories,curr_generic=None,curr_categories=None,prev_vectors=None,term = None, chamber = None,min_term=111,n_terms=9):
  if not prev_vectors:
    prev_vectors={"cum_prev_generics":np.zeros(len(all_generics)),
                  "recent_generics":np.zeros(len(all_generics)),
                  "generic":np.zeros(len(all_generics)),
                  "cum_prev_categories":np.zeros(len(all_categories)),
                  "recent_categories":np.zeros(len(all_categories)),
                  "categories":np.zeros(len(all_categories))}
  generics_inv = {name:i for i,name in enumerate(all_generics)}
  categories_inv = {name:i for i,name in enumerate(all_categories)}
  generic_vec = np.zeros(len(all_generics))
  categories_vec = np.zeros(len(all_categories))
  if curr_generic and curr_generic in generics_inv:
    generic_vec[generics_inv[curr_generic]]=1
  if curr_categories:
    for category in curr_categories:
      categories_vec[categories_inv[category]]=1
  chamber_vec = np.zeros(2)
  if chamber=="House":
    chamber_vec[0]=1
  elif chamber=="Senate":
    chamber_vec[1]=1
  term_vec=np.zeros(n_terms)
  if term:
    term_vec[term-min_term]=1
  return {
    "recent_generics":np.array(prev_vectors["generic"]*(1-decay)+decay*prev_vectors["recent_generics"]),
    "cum_prev_generics":np.array(prev_vectors["generic"]+prev_vectors["cum_prev_generics"]),
    "recent_categories":np.array(prev_vectors["categories"]*(1-decay)+decay*prev_vectors["recent_categories"]),
    "cum_prev_categories":np.array(prev_vectors["categories"]+prev_vectors["cum_prev_categories"]),
    "term":term_vec,
    "chamber":chamber_vec,
    "generic":generic_vec,
    "categories":categories_vec
  }
def create_vectors_bill_unnormalized(bill_df,all_generics,all_categories,decay): 
  out=pd.DataFrame(columns=["recent_generics","cum_prev_generics","recent_categories","cum_prev_categories","output_generic","output_categories"])
  vecs = None
  for i,row in bill_df.iterrows():
    vecs=create_next_vector_unnormalized(decay,all_generics,all_categories,row["generic"],row["categories"],vecs,term=row["term"],chamber=row["chamber"])
    out.loc[len(out)]=vecs
  return out

def create_next_vector(std_generics,std_categories,**kwargs):
  out = create_next_vector_unnormalized(**kwargs)
  out = normalize(out,std_generics,std_categories)
  return out

def normalize(to_normalize,std_generics,std_categories):
  to_normalize["cum_prev_generics"] = list(np.stack(to_normalize["cum_prev_generics"],axis=0) /std_generics[None,:])
  to_normalize["cum_prev_catgories"] = list(np.stack(to_normalize["cum_prev_categories"],axis=0) /std_categories[None,:])
  return to_normalize

if __name__=="__main__":

  parser = argparse.ArgumentParser(description="prepare data with generics")
  parser.add_argument("-d","--preprocessing_dir",type=str,default="./outputs/preprocess0", help="the directory for this preprocessing run")
  parser.add_argument("--decay_factor",type=float,default=2/3,help="the factor by which the previous generics decay for recent_generics vectors")
  parser.add_argument("--common_threshold",type=float,default=200,help="the min number of instances for a generic or category to be considered common")
  args,unknown = parser.parse_known_args()

  with open(os.path.join(args.preprocessing_dir,"generics_dict_manual.json"),"r") as file:
    generics_dict = json.load(file)
  with open(os.path.join(args.preprocessing_dir,"categories_dict.json"),"r") as file:
    categories_dict = json.load(file)
  common_generics = {generic for generic in generics_dict if len(generics_dict[generic])>=args.common_threshold}
  common_generics.add("Miscellaneous")
  common_categories = {category for category in categories_dict if len(categories_dict[category])>=args.common_threshold}

  data = pd.read_csv(os.path.join(args.preprocessing_dir,"initial_data.csv"))
  generic_map={}
  for generic in generics_dict:
    if generic not in common_generics:
      continue
    for action in generics_dict[generic]:
      generic_map[action]=generic
  for action in set(data["action"]):
    if action not in generic_map:
      generic_map[action]="Miscellaneous"
  category_map=defaultdict(list)
  for category in categories_dict:
    if category not in common_categories:
      continue
    for action in categories_dict[category]:
      category_map[action].append(category)
  data["generic"]=data["action"].apply(lambda x:generic_map[x])
  data["categories"]=data["action"].apply(lambda x:category_map[x])

  bills = [bill for (id, bill) in data.groupby("bill_id")]
  random.seed(41)
  random.shuffle(bills)
  train_data = pd.concat(bills[:int(0.8*len(bills))])
  test_data = pd.concat(bills[int(0.8*len(bills)):])
  all_data = pd.concat(bills)

  common_generics=list(common_generics)
  common_categories=list(common_categories)
  with open(os.path.join(args.preprocessing_dir,"generics.json"),"w") as file:
    json.dump(common_generics,file)
  with open(os.path.join(args.preprocessing_dir,"categories.json"),"w") as file:
    json.dump(common_categories,file)
  train_vecs = pd.concat([create_vectors_bill_unnormalized(bill,common_generics,common_categories,args.decay_factor) for i,bill in train_data.groupby("bill_id")])
  std_generics = np.std(np.stack(train_vecs["cum_prev_generics"],axis=0),axis=0)
  std_categories = np.std(np.stack(train_vecs["cum_prev_categories"],axis=0),axis=0)
  train_vecs=normalize(train_vecs,std_generics,std_categories)
  test_vecs = normalize(pd.concat([create_vectors_bill_unnormalized(bill,common_generics,common_categories,args.decay_factor) for i,bill in test_data.groupby("bill_id")]),std_generics,std_categories)

  with open(os.path.join(args.preprocessing_dir,"train_vectors.pkl"),"wb") as file:
    train_vecs.to_pickle(file)
  with open(os.path.join(args.preprocessing_dir,"test_vectors.pkl"),"wb") as file:
    test_vecs.to_pickle(file)
  

