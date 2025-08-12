import pandas as pd
from collections import defaultdict
import random
import json


with open("../outputs/generics_08-04_manual-llm-manual.json","r") as file:
  generics = json.load(file)
with open("../outputs/categories_08-04.json","r") as file:
  categories = json.load(file)
generic_map={}
for name in generics:
  for action in generics[name]:
    generic_map[action]=name
category_map=defaultdict(list)
for name in categories:
  for i,action in enumerate(categories[name]):
    category_map[action].append(name)

data = pd.read_csv("../outputs/data/data_pregenericing.csv")
data["generic"]=data["action"].apply(lambda x:generic_map[x])
data["categories"]=data["action"].apply(lambda x:category_map[x])
bills = [bill for (id, bill) in data.groupby("bill_id")]
random.seed(41)
random.shuffle(bills)
train_data = pd.concat(bills[:int(0.8*len(bills))])
test_data = pd.concat(bills[int(0.8*len(bills)):])
all_data = pd.concat(bills)
train_data.to_csv("../outputs/data/train_data.csv")
test_data.to_csv("../outputs/data/test_data.csv")
all_data.to_csv("../outputs/data/all_data.csv")
