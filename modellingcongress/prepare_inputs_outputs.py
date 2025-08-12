import numpy as np
import json
import pandas as pd
import sklearn


with open("../outputs/generics_08-04_manual-llm-manual.json","r") as file:
  generics = json.load(file)
with open("../outputs/categories_08-04.json","r") as file:
  categories = json.load(file)
common_generic_names = [name for name in generics if len(generics[name])>=10]
common_category_names = [name for name in categories if len(categories[name])>=10]
with open("../outputs/common_generic_names.json","w") as file:
  json.dump(common_generic_names,file)
with open("../outputs/common_category_names.json","w") as file:
  json.dump(common_category_names,file)
with open("../outputs/common_generic_names.json","r") as file:
  common_generic_names = json.load(file)
with open("../outputs/common_category_names.json","r") as file:
  common_category_names = json.load(file)
common_generic_names_inv = {name:i for i,name in enumerate(common_generic_names)}
common_category_names_inv = {name:i for i,name in enumerate(common_generic_names)}

MIN_TERM=111
N_TERMS = 9
alpha=2/3

# creates vectors outto use as input and output of the model from a certain bill
# predecessor is an exponentially weighted moving average of the one-hot
# vector representing the previous generic
# predecessors is a sum of all one-hot vectors representing previous actions 
def create_vectors_bill(bill_df):
  out=[]
  predecessors_generics=np.zeros(len(common_generic_names))
  predecessor_generics=np.zeros(len(common_generic_names))
  predecessors_categories=np.zeros(len(common_category_names))
  predecessor_categories=np.zeros(len(common_category_names))
  curr_generic = np.zeros(len(common_generic_names))
  curr_categories = np.zeros(len(common_category_names))
  for i,row in bill_df.iterrows():
    prev_generic=np.array(curr_generic)
    prev_categories=np.array(curr_categories)
    curr_generic = np.zeros(len(common_generic_names))
    curr_categories = np.zeros(len(common_category_names))
    output_generic = np.zeros(len(common_generic_names)+1)
    if row["generic"] in common_generic_names_inv:
      curr_generic[common_generic_names_inv[row.generic]]=1
      output_generic[common_generic_names_inv[row.generic]]=1
    else:
      output_generic[-1]=1
    if row["categories"]:
      for category in row["categories"]: 
        if category in common_category_names_inv:
          curr_categories[common_category_names_inv[category]]=1
    
    predecessor_generics=(1-alpha)*np.array(predecessor_generics)+alpha*np.array(prev_generic)
    predecessors_generics=np.array(predecessors_generics)+prev_generic
  

    predecessor_categories=(1-alpha)*predecessor_categories+alpha*np.array(prev_categories)
    predecessors_categories=np.array(predecessors_categories)+prev_categories
    chamber = np.zeros(2)
    if row["chamber"]=="House":
      chamber[0]=1
    elif row["chamber"]=="Senate":
      chamber[1]=1
    term=np.zeros(N_TERMS)
    if row["term"]:
      term[row["term"]-MIN_TERM]=1
    entry={"predecessor_generics":predecessor_generics,"predecessors_generics":predecessors_generics,"predecessor_categories":predecessor_categories,"predecessors_categories":predecessors_categories,"term":term,"chamber":chamber,
          "output_generic":output_generic,"output_categories":curr_categories} # print(entry)
    out.append(entry)
  return out
# concatenate a list of lists
def concat(ls):
  out=[]
  for l in ls:
    out+=l
  return out
bills = pd.read_csv("../outputs/data/train_data.csv").groupby("bill_id")
data = pd.DataFrame(concat(create_bill_vectors(bill) for i,bill in bills))
preds_generics = np.stack(data["predecessors_generics"],axis=0)
preds_categories = np.stack(data["predecessors_categories"],axis=0)
scaler_preds_generics = sklearn.preprocessing.StandardScaler().fit(preds_generics)
scaler_preds_categories = sklearn.preprocessing.StandardScaler().fit(preds_categories)

# creates bill vectors but with the predecessors axes normalized
# so that coefficients are of similar size to predecessor
def create_vectors_bill_normalized(bill):
  out = create_vectors_bill(bill)
  for d in out:
    d["predecessors_generics"] = scaler_preds_generics.transform([d["predecessors_generics"]])[0]
    d["predecessors_categories"] = scaler_preds_categories.transform([d["predecessors_categories"]])[0]
  return out


if __name__=="__main__":
  # manually scale the existing vectors so you don't have to recalculate all the vectors
  data["predecessors_generics"]=scaler_preds_generics.transform(preds_generics)
  data["predecessors_categories"]=scaler_preds_categories.transform(preds_categories)
  with open("../outputs/prediction_vecs","wb") as file:
    data.to_pickle(file)
