import pandas as pd
from collections import defaultdict
import random
import json
import os
import argparse


parser = argparse.ArgumentParser(description="prepare data with generics")
parser.add_argument("-d","--preprocessing_dir",type=str,default="./outputs/preprocess0", help="the directory for this preprocessing run")
args,unknown = parser.parse_known_args()

with open(os.path.join(args.preprocessing_dir,"generics_dict_manual_llm_manual.json"),"r") as file:
  generics_dict = json.load(file)
with open(os.path.join(args.preprocessing_dir,"categories_dict.json"),"r") as file:
  categories = json.load(file)
generic_map={}
for generic in generics_dict:
  for action in generics_dict[generic]:
    generic_map[action]=generic
category_map=defaultdict(list)
for name in categories:
  for i,action in enumerate(categories[name]):
    category_map[action].append(name)


data = pd.read_csv(os.path.join(args.preprocessing_dir,"data_no_generics.csv"))
data["generic"]=data["action"].apply(lambda x:generic_map[x])
data["categories"]=data["action"].apply(lambda x:category_map[x])
bills = [bill for (id, bill) in data.groupby("bill_id")]
random.seed(41)
random.shuffle(bills)
train_data = pd.concat(bills[:int(0.8*len(bills))])
test_data = pd.concat(bills[int(0.8*len(bills)):])
all_data = pd.concat(bills)

train_data.to_csv(os.path.join(args.output_dir,"train_data.csv"))
test_data.to_csv(os.path.join(args.output_dir,"test_data.csv"))
all_data.to_csv(os.path.join(args.output_dir,"all_data.csv"))
