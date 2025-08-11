
import torch
from torch.utils.data import DataLoader
from torch.nn.functional import softmax,sigmoid
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


client = openai.Client(api_key=os.environ["OPENAI_API_KEY"])
with open("outputs/buckets_08-04_manual-llm-manual.json") as file:
  buckets = json.load(file)

def refine_actions(actions):
  action_batches = np.split(actions,range(5,len(actions),5))
  refinements=list(actions)
  action_i=0
  for batch in action_batches:
      inpt = llm_input(batch)
      responses = [x for x in client.responses.create(model="gpt-5-mini",input=inpt).output_text.split("\n") if x!=""]
      for i,action in enumerate(batch):
        j=min(i,len(responses)-1)
        for k,response in enumerate([responses[j]]+responses[j+1:len(responses)]+responses[:j]):
          if edit_distance_below(response,action,1/2*max(len(response),len(action))):
            refinements[action_i]=response
            break
        action_i+=1
  return refinements

def bucket_refined_actions(refined_actions):
  out=[None for i in range(len(refined_actions))]
  for i,refinement in enumerate(refined_actions):
    for name in reversed(buckets):
      if edit_distance_below(refinement,name,1/7*max(len(name),len(refinement))):
        out[i]=(name)
        break
  return out



 


with open("outputs/common_bucket_names.json","r") as file:
  common_bucket_names = json.load(file)
with open("outputs/common_extra_bucket_names.json","r") as file:
  common_extra_bucket_names = json.load(file)
def predict_bill(bill_df,refine_first=True):
  if refine_first:
    refined_actions = refine_actions(bill_df["action"])
  bill_df["bucket"]=bucket_refined_actions(refined_actions)
  bill_df["extra_buckets"]=bill_df["action"].apply(extra_buckets_map)
  bill_df.loc[-1]=([None for i in range(len(bill_df.columns))])
  
  vecs = create_bill_vectors(bill_df)

  state_dict = torch.load("outputs/models/08-04_withextras_bce_lr1e-5_beta1e-06/epoch200.pt")
  weights = state_dict["model"]["weight"]
  model=torch.nn.Linear(weights.shape[1],weights.shape[0])
  dotenv.load_dotenv()

  ds=ActionDataset(vecs)
  loader = DataLoader(ds,batch_size= None,shuffle=False)
  out=[]
  with torch.no_grad():
    for i,(inpt, output) in enumerate(loader):
      inpt=inpt.float()
      pred = model(inpt)
      pred[:len(common_bucket_names)+1]=softmax(pred[:len(common_bucket_names)+1])
      pred[len(common_bucket_names)+1:]=sigmoid(pred[len(common_bucket_names)+1:])
      
      p_misc=pred[len(common_bucket_names)+1]
      pred_buckets = sorted([(p/(1-p_misc),b) for p,b in zip(pred[:len(common_bucket_names)],common_bucket_names)],reverse=True)
      pred_extra_buckets = sorted([(p,b) for p,b in zip(pred[len(common_extra_bucket_names)+1:],common_extra_bucket_names)],reverse=True)
      p_real_action=[p for p,b in pred_buckets if b==bill_df.iloc[i]["bucket"]]
  out.append(pred_buckets)
  return out

with open("outputs/data/test_data.csv","r") as file:
  bill_dfs = [x[1] for x in pd.read_csv(file).groupby("bill_id")]
  bill_dfs.sort(key=lambda x:len(x["action"]),reverse=True)

predictions = predict_bill(bill_dfs[7000])
