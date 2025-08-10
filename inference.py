import torch
from torch.utils.data import DataLoader
import json
import dotenv
import openai
import prediction_model
from llm_refine_bucket_names import llm_input
from prediction_model import create_bill_vectors,ActionDataset
from manual_bucketing import edit_distance_below
import openai
import numpy as np
import os
import pandas as pd
from extra_bucketing import extra_buckets_map
state_dict = torch.load("outputs/models/08-04_withextras_bce_lr1e-5_beta1e-06/epoch200.pt")
weights = state_dict["model"]["weight"]
model=torch.nn.Linear(weights.shape[1],weights.shape[0])
dotenv.load_dotenv()
client = openai.Client(api_key=os.environ["OPENAI_API_KEY"])
with open("outputs/buckets_08-04_manual-llm-manual.json") as file:
  buckets = json.load(file)

def bucket(action):
  input = llm_input([action])
  refinement = client.responses.create(model="gpt-5",input=input).output_text
  print("refinement:",refinement)
  for name in buckets:
    if edit_distance_below(refinement,name,1/7*max(len(name),len(refinement))):
      return name
  return None
 


with open("outputs/common_bucket_names.json","r") as file:
  common_bucket_names = json.load(file)
with open("outputs/common_extra_bucket_names.json","r") as file:
  common_extra_bucket_names = json.load(file)
def predict_bill(bill_df):
  bill_df["bucket"]=bill_df["action"].apply(bucket)
  bill_df["extra_buckets"]=bill_df["action"].apply(extra_buckets_map)
  bill_df.loc[-1]=([None for i in range(len(bill_df.columns))])
  vecs = create_bill_vectors(bill_df)
  print("created bill vectors")


  ds=ActionDataset(vecs)
  loader = DataLoader(ds,batch_size= None,shuffle=False)
  print("created loader")
  for i,(inpt, output) in enumerate(loader):
    pred = model(input)
    print("input:",bill_df.iloc[:i])
    pred_buckets = sorted([(p,bucket) for p in zip(pred[:len(common_bucket_names)],common_bucket_names)],reverse=True)
    print("pred buckets:",pred_buckets)

with open("outputs/data/test_data.csv","r") as file:
  bill_dfs = [x[1] for x in pd.read_csv(file).groupby("bill_id")]
  bill_dfs.sort(key=lambda x:len(x["action"]),reverse=True)
# display(bill_dfs[7000])
predict_bill(bill_dfs[7000])
# for i,bill_df in bill_dfs:
#   predict_bill(bill_df)
#   break

