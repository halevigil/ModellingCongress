import pandas as pd
from collections import defaultdict
import random
import json
import os
import argparse
import numpy as np
import sklearn
import pickle



# Class for creating input and output vectors
class CreateInputsOutputs():
  def __init__(self,inference_dir):
    self.inference_dir=inference_dir
    with open(os.path.join(inference_dir,"generics.json"),"r") as file:
      self.generics = json.load(file)
    with open(os.path.join(inference_dir,"categories.json"),"r") as file:
      self.categories = json.load(file)
    with open(os.path.join(inference_dir,"processing.json"),"r") as file:
      self.processing = json.load(file)
    if os.path.exists(os.path.join(inference_dir,"scale_factors.npy")):
      self.scale_factors = np.load(os.path.join(inference_dir,"scale_factors.npy"))
    self.generics_inv = {x:i for i,x in enumerate(self.generics)}
    self.categories_inv = {x:i for i,x in enumerate(self.categories)}

  def get_generics(self):
    return self.generics
  
  def get_categories(self):
    return self.categories
  def input_length(self):
    return 2*len(self.generics)+2*len(self.categories)+2+self.processing["n_terms"]
  def output_length(self):
    return len(self.generics)+len(self.categories)
  # creates output vector for an action
  def create_output_vector(self,generic=None,categories=None):
    generic_vec = np.zeros(len(self.generics))
    categories_vec = np.zeros(len(self.categories))
    if generic and generic in self.generics_inv:
      generic_vec[self.generics_inv[generic]]=1
    if categories:
      for category in categories:
        if category in self.categories:
          categories_vec[self.categories_inv[category]]=1
    return np.concatenate([generic_vec,categories_vec])
  # creates input vector for an action
  def create_input_vector_unnormalized(self,prev_input_vector=None,prev_output_vector=None,term=None,chamber=None,min_term=111,n_terms=9):
    if prev_input_vector is None:
      prev_input_vector=np.zeros(2*len(self.generics)+2*len(self.generics)+2+n_terms)
    if prev_output_vector is None:
      prev_output_vector=np.zeros(len(self.generics)+len(self.categories))
    
    chamber_vec = np.zeros(2)
    if chamber=="House":
      chamber_vec[0]=1
    elif chamber=="Senate":
      chamber_vec[1]=1
    term_vec=np.zeros(n_terms)
    if term:
      term_vec[int(term-min_term)]=1
    
    prev_recent_generics=prev_input_vector[:len(self.generics)]
    prev_cumulative_generics=prev_input_vector[len(self.generics):2*len(self.generics)]
    prev_recent_categories=prev_input_vector[2*len(self.generics):2*len(self.generics)+len(self.categories)]
    prev_cumulative_categories=prev_input_vector[2*len(self.generics)+len(self.categories):2*len(self.generics)+2*len(self.categories)]
    prev_output_generics = prev_output_vector[:len(self.generics)]
    prev_output_categories = prev_output_vector[len(self.generics):]
    

    out_dict= {
      "recent_generics":np.array(prev_output_generics*(1-self.processing["decay"])+self.processing["decay"]*prev_recent_generics),
      "cumulative_generics":np.array(prev_output_generics+prev_cumulative_generics),
      "recent_categories":np.array(prev_output_categories*(1-self.processing["decay"])+self.processing["decay"]*prev_recent_categories),
      "cumulative_categories":np.array(prev_output_categories+prev_cumulative_categories),
      "term":term_vec,
      "chamber":chamber_vec
    }
    return np.concatenate([out_dict["recent_generics"],out_dict["cumulative_generics"],out_dict["recent_categories"],out_dict["cumulative_categories"],out_dict["term"],out_dict["chamber"]])

  # creates input vector normalized
  def create_input_vector(self,prev_input_vector=None,prev_output_vector=None,term=None,chamber=None,min_term=111,n_terms=9):
    out=self.create_input_vector_unnormalized(prev_input_vector,prev_output_vector,term,chamber)/self.scale_factors
    return out
  
  # create vectors for all rows in the dataset
  # these vectors are calculated by bill
  def vectors_by_bill(self,df):
    bills = [bill for i,bill in df.groupby("bill_id")]
    input_vec=None
    output_vec=None
    input_vecs=[]
    output_vecs=[]
    for bill in bills:
      for i,row in bill.iterrows():
        input_vec = self.create_input_vector_unnormalized(input_vec,output_vec,row["term"],row["chamber"],self.processing["min_term"],self.processing["n_terms"])
        output_vec = self.create_output_vector(row["generic"],row["categories"])
        input_vecs.append(input_vec)
        output_vecs.append(output_vec)
    input_mat = np.stack(input_vecs)
    stds=np.std(input_mat,axis=0)
    self.scale_factors=np.ones_like(input_vec)
    self.scale_factors[len(self.generics):len(self.generics)*2]=stds[len(self.generics):len(self.generics)*2]
    self.scale_factors[2*len(self.generics)+len(self.categories):len(self.generics)*2+2*len(self.categories)]=stds[2*len(self.generics)+len(self.categories):len(self.generics)*2+2*len(self.categories)]

    input_vecs = [vec/self.scale_factors for vec in input_vecs]
    np.save(os.path.join(inference_dir,"scale_factors"),self.scale_factors)
    return input_vecs,output_vecs
    
  def vector_to_probabilities(self,vecs):
    return {self.generics[i]:float(vecs[i]) for i in range(len(self.generics))},{self.categories[i]:vecs[i+len(self.generics)] for i in range(len(self.categories))}
if __name__=="__main__":

  parser = argparse.ArgumentParser(description="prepare data with generics")
  parser.add_argument("-d","--preprocessing_dir",type=str,default="outputs/preprocess1", help="the directory for this preprocessing run")
  parser.add_argument("-i","--inference_dir",default=None,type=str, help="the directory for the data required for inference.defaults to preprocessing_dir/inference")
  
  parser.add_argument("--decay_factor",type=float,default=2/3,help="the factor by which the previous generics decay for recent_generics vectors")
  parser.add_argument("--common_threshold",type=float,default=200,help="the min number of instances for a generic or category to be considered common")
  args,unknown = parser.parse_known_args()

  with open(os.path.join(args.preprocessing_dir,"generics_dict_manual_llm_manual.json"),"r") as file:
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

  common_generics=list(common_generics)
  common_categories=list(common_categories)
  inference_dir = args.inference_dir or os.path.join(args.preprocessing_dir,"inference")
  if not os.path.exists(inference_dir):
    os.mkdir(inference_dir)
  with open(os.path.join(inference_dir,"generics.json"),"w") as file:
    json.dump(common_generics,file)
  with open(os.path.join(inference_dir,"categories.json"),"w") as file:
    json.dump(common_categories,file)
  with open(os.path.join(inference_dir,"processing.json"),"w") as file:
    json.dump({"min_term":111,"n_terms":9,"decay":2/3},file)

  input_output_creator = CreateInputsOutputs(inference_dir)
  input_vecs,output_vecs = input_output_creator.vectors_by_bill(data)
  input_vecs=np.stack(input_vecs,axis=0)
  output_vecs=np.stack(output_vecs,axis=0)
  np.save(os.path.join(args.preprocessing_dir,"input_vectors"),input_vecs)
  np.save(os.path.join(args.preprocessing_dir,"output_vectors"),output_vecs)

  

