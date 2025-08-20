
import torch
from torch.utils.data import DataLoader
from torch.nn.functional import softmax,sigmoid
import json
import dotenv
import openai
from llm_refinement import llm_input
from prediction_model import ActionDataset
from make_generics import edit_distance_below
from prepare_inputs_outputs import CreateInputsOutputs
from categorize import categorize
import openai
import numpy as np
import os
import pandas as pd
from categorize import categorize
from llm_refinement import create_refinements_nobatch


client = openai.Client(api_key=os.environ["OPENAI_API_KEY"])

def get_generic(actions,generics):
  out=["Miscellaneous" for i in range(len(actions))]
  for i,refinement in enumerate(actions):
    for name in reversed(generics):
      if edit_distance_below(refinement,name,1/7*max(len(name),len(refinement))):
        out[i]=(name)
        break
  return out
def predict_action_from_seq(model,action,inference_dir,prev_input_vector=None,need_refinement=False,need_to_make_generic=False,chamber=None,term=None):
  input_output_creator = CreateInputsOutputs(inference_dir)
  if need_refinement:
    refined_action=create_refinements_nobatch([action])[0]
  else:
    refined_action=action
  if need_to_make_generic:
    generic = get_generic([refined_action],input_output_creator.get_generics())[0]
  else:
    generic=action
  categories = categorize(action)
  prev_output_vectors=input_output_creator.create_output_vector(generic,categories)
  input_vector = torch.from_numpy(input_output_creator.create_input_vector(prev_input_vector=prev_input_vector,prev_output_vector=input_output_creator.create_output_vector(generic,categories),chamber=chamber,term=term)).float()
  with torch.no_grad():
    output=model(input_vector)
  output[:len(input_output_creator.get_generics())]=torch.nn.functional.softmax(output[:len(input_output_creator.get_generics())],dim=0)
  output[len(input_output_creator.get_generics()):]=torch.nn.functional.sigmoid(output[len(input_output_creator.get_generics()):])
  probabilities=input_output_creator.vector_to_probabilities(output)
  return probabilities,input_vector
def predict_action_from_last(model,action,inference_dir,prev_input_vectors=None,need_refinement=False,need_to_make_generic=False,chamber=None,term=None):
  input_output_creator = CreateInputsOutputs(inference_dir)
  if need_refinement:
    refined_action=create_refinements_nobatch([action])[0]
  else:
    refined_action=action
  if need_to_make_generic:
    generic = get_generic([refined_action],input_output_creator.get_generics())[0]
  else:
    generic=action
  categories = categorize(action)
  prev_output_vectors=input_output_creator.create_output_vector(generic,categories)
  input_vectors = torch.from_numpy(input_output_creator.create_input_vector(prev_input_vector=prev_input_vectors,prev_output_vector=input_output_creator.create_output_vector(generic,categories),chamber=chamber,term=term)).float()
  with torch.no_grad():
    output=model(input_vectors)
  output[:len(input_output_creator.get_generics())]=torch.nn.functional.softmax(output[:len(input_output_creator.get_generics())],dim=0)
  output[len(input_output_creator.get_generics()):]=torch.nn.functional.sigmoid(output[len(input_output_creator.get_generics()):])
  probabilities=input_output_creator.vector_to_probabilities(output)
  return probabilities
def load_model(model_path):
  state_dict = torch.load(model_path)
  model=torch.nn.Linear(state_dict["model"]["weight"].shape[1],state_dict["model"]["weight"].shape[0])
  return model
predicted_generics,predicted_categories = predict_action_from_last(load_model("outputs/preprocess0/models/lr3e-04_lassoweight1e-05_batch256/epoch120.pt"),action="Cloture motion on the motion to proceed to measure presented in Senate.",inference_dir="outputs/preprocess0/inference")
display(sorted(predicted_categories.items(),key=lambda x:x[1],reverse=True))
  

# def predict_bill(bill_df,refine_first=True):
#   if refine_first:
#     refined_actions = refine_actions(bill_df["action"])
#   bill_df["generic"]=generic_refined_actions(refined_actions)
#   bill_df["categories"]=bill_df["action"].apply(categories_map)
#   bill_df.loc[-1]=([None for i in range(len(bill_df.columns))])
  
#   vecs = create_bill_vectors(bill_df)

#   state_dict = torch.load("./outputs/models/08-04_withextras_bce_lr1e-5_beta1e-06/epoch200.pt")
#   weights = state_dict["model"]["weight"]
#   model=torch.nn.Linear(weights.shape[1],weights.shape[0])
#   dotenv.load_dotenv()

#   ds=ActionDataset(vecs)
#   loader = DataLoader(ds,batch_size= None,shuffle=False)
#   out=[]
#   with torch.no_grad():
#     for i,(inpt, output) in enumerate(loader):
#       inpt=inpt.float()
#       pred = model(inpt)
#       pred[:len(common_generic_names)+1]=softmax(pred[:len(common_generic_names)+1])
#       pred[len(common_generic_names)+1:]=sigmoid(pred[len(common_generic_names)+1:])
      
#       p_misc=pred[len(common_generic_names)+1]
#       pred_generics = sorted([(p/(1-p_misc),b) for p,b in zip(pred[:len(common_generic_names)],common_generic_names)],reverse=True)
#       pred_categories = sorted([(p,b) for p,b in zip(pred[len(common_category_names)+1:],common_category_names)],reverse=True)
#       p_real_action=[p for p,b in pred_generics if b==bill_df.iloc[i]["generic"]]
#   out.append(pred)
#   return out

# with open("./outputs/data/test_data.csv","r") as file:
#   bill_dfs = [x[1] for x in pd.read_csv(file).groupby("bill_id")]
#   bill_dfs.sort(key=lambda x:len(x["action"]),reverse=True)

# predictions = predict_bill(bill_dfs[7000])
