import torch
import json
import dotenv
import openai
from llm_refine_bucket_names import llm_input
from prediction_model_analysis import test_dataset
from manual_bucketing import edit_distance_below
import openai
import numpy as np
import os
state_dict = torch.load("outputs/models/08-04_withextras_bce_lr1e-5_beta1e-06/epoch200.pt")
weights = state_dict["model"]["weight"]
model=torch.nn.Linear(weights.shape[1],weights.shape[0])
dotenv.load_dotenv()
client = openai.Client(api_key=os.environ["OPENAI_API_KEY"])
with open("outputs/buckets_08-04_manual-llm-manual.json") as file:
  buckets = json.load(file)

def predict(actions,chambers=None,):
  actions_df=[]
  for action_batch in np.split(actions,range(5,len(actions),5)):
    input = llm_input(action_batch)
    refined_action = client.responses.create(model="gpt-5",input=input).output_text
    for line in refined_action.split("\n"):
      if line=="":
        continue
      for name in buckets:
        if edit_distance_below(line,refined_action,1/7*max(len(name),len(refined_action))):
          buckets.append({"bucket":name})
          bucketed=True
      



  

