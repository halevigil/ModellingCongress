# Prepares input and output vectors for the prediction model
import numpy as np
import json
import pandas as pd
import sklearn
import argparse
import os

parser = argparse.ArgumentParser(description="prepares input and output vectors for the prediction model")
parser.add_argument("-d","--preprocessing_dir",type=str,default="./outputs/preprocess0", help="the directory for this preprocessing run")
parser.add_argument("--decay_factor",type=float,default=2/3,help="the factor by which the previous generics decay for recent_generics vectors")

args,unknown = parser.parse_known_args()


MIN_TERM=111
N_TERMS = 9
# creates vectors to use as input and output of the model from a certain bill
# recent is an exponentially weighted moving average of the one-hot
# vector representing the previous generic
# cum_prev is a sum of all one-hot vectors representing previous actions 
# the outputs are a one-hot representation of the current generic
# and a binary for each category
def create_vectors_bill(bill_df,common_generics,common_categories,alpha):
  common_generics_inv = {name:i for i,name in enumerate(common_generics)}
  common_categories_inv = {name:i for i,name in enumerate(common_generics)}

  out=pd.DataFrame(columns=["recent_generics","cum_prev_generics","recent_categories","cum_prev_categories","output_generic","output_categories"])
  cum_prev_generics=np.zeros(len(common_generics))
  recent_generics=np.zeros(len(common_generics))
  cum_prev_categories=np.zeros(len(common_categories))
  recent_categories=np.zeros(len(common_categories))
  curr_generic = np.zeros(len(common_generics))
  curr_categories = np.zeros(len(common_categories))
  for i,row in bill_df.iterrows():
    prev_generic=np.array(curr_generic)
    prev_categories=np.array(curr_categories)
    curr_generic = np.zeros(len(common_generics))
    curr_categories = np.zeros(len(common_generics))
    output_generic = np.zeros(len(common_generics)+1)
    if row["generic"] in common_generics_inv:
      curr_generic[common_generics_inv[row.generic]]=1
      output_generic[common_generics_inv[row.generic]]=1
    else:
      output_generic[-1]=1
    if row["categories"]:
      for category in row["categories"]: 
        if category in common_categories_inv:
          curr_categories[common_categories_inv[category]]=1
    
    recent_generics=alpha*np.array(recent_generics)+alpha*np.array(prev_generic)
    cum_prev_generics=np.array(cum_prev_generics)+prev_generic
  

    recent_categories=alpha*recent_categories+(1-alpha)*np.array(prev_categories)
    cum_prev_categories=np.array(cum_prev_categories)+prev_categories
    chamber = np.zeros(2)
    if row["chamber"]=="House":
      chamber[0]=1
    elif row["chamber"]=="Senate":
      chamber[1]=1
    term=np.zeros(N_TERMS)
    if row["term"]:
      term[row["term"]-MIN_TERM]=1
    out.loc[-1]=[recent_generics,cum_prev_generics,recent_categories,cum_prev_categories,term,chamber,output_generic,curr_categories]
  return out
# concatenate a list of lists
def concat(ls):
  out=[]
  for l in ls:
    out+=l
  return out
# Normalize the cum_prev axes to mean=0 std=0.5
# so that coefficients are of similar size to recent
# inplace and also return updated vectors
def normalize(to_normalize,all_data):
  scaler_preds_generics = sklearn.preprocessing.StandardScaler().fit(all_data["cum_prev_generics"])
  scaler_preds_categories = sklearn.preprocessing.StandardScaler().fit(all_data["cum_prev_categories"])

  to_normalize["cum_prev_generics"] = scaler_preds_generics.transform(to_normalize["cum_prev_generics"])
  to_normalize["cum_prev_categories"] = scaler_preds_categories.transform([to_normalize["cum_prev_categories"]])[0]
  return to_normalize
# creates bill vectors with cum_prev normalized
def create_vectors_bill_normalized(bill,data,common_generics,common_categories,alpha):
  out = create_vectors_bill(bill)
  normalize(out,data)
  return out

if __name__=="__main__":
  with open(os.path.join(args.preprocessing_dir,"generics_dict_manual_llm_manual.json"),"r") as file:
    generics_dict = json.load(file)
  with open(os.path.join(args.preprocessing_dir,"categories_dict.json"),"r") as file:
    categories_dict = json.load(file)
  common_generics = [generic for generic in generics_dict if len(generics_dict[generic])>=100]
  common_categories = [category for category in categories_dict if len(categories_dict[category])>=100]
  with open(os.path.join(args.preprocessing_dir,"common_generics.json"),"w") as file:
    json.dump(common_generics,file)
  with open(os.path.join(args.preprocessing_dir,"common_categories.json"),"w") as file:
    json.dump(common_categories,file)


  bills = pd.read_csv(os.path.join(args.preprocessing_dir,"all_data.csv")).groupby("bill_id")
  data = pd.DataFrame(concat(create_vectors_bill(bill,common_generics,common_categories,args.decay_factor) for i,bill in bills))
  with open(os.path.join(args.preprocessing_dir,"all_vectors_unnormalized.pkl"),"wb") as file:
    data.to_pickle(file)
  
  train_bills = pd.read_csv(os.path.join(args.preprocessing_dir,"train_data.csv")).groupby("bill_id")
  train_data = pd.DataFrame(concat(create_vectors_bill_normalized(bill,data,common_generics,common_categories,args.decay_factor) for i,bill in train_bills))
  with open(os.path.join(args.preprocessing_dir,"train_vectors.pkl"),"wb") as file:
    data.to_pickle(file)
  
  test_bills = pd.read_csv(os.path.join(args.preprocessing_dir,"test_data.csv")).groupby("bill_id")
  test_data = pd.DataFrame(concat(create_vectors_bill_normalized(bill,data,common_generics,common_categories,args.decay_factor) for i,bill in train_bills))
  with open(os.path.join(args.preprocessing_dir,"test_vectors.pkl"),"wb") as file:
    data.to_pickle(file)
  
