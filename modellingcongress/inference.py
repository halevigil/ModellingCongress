import json
import dotenv
import openai
import onnx
from scipy.special import softmax,expit
from IPython.display import display

import onnxruntime as ort

if __package__:
  from .llm_refinement import llm_input
else:
  from llm_refinement import llm_input


if __package__:
    # Relative imports for package usage
  from .make_generics import edit_distance_below
  from .prepare_inputs_outputs import CreateInputsOutputs
  from .categorize import categorize
  from .llm_refinement import create_refinements
else:
  from make_generics import edit_distance_below
  from prepare_inputs_outputs import CreateInputsOutputs
  from categorize import categorize
  from llm_refinement import create_refinements
import openai
import numpy as np
import os
import pandas as pd
# from llm_refinement import create_refinements
dotenv.load_dotenv()


client = openai.Client(api_key=os.environ["OPENAI_API_KEY"])

def make_generic(action,generics):
  for name in reversed(generics):
    if edit_distance_below(action,name,1/7*max(len(name),len(action))):
      return action
  return "Miscellaneous"
def predict_action_from_seq(model_name,prev_actions,inference_dir,prev_input_vector = None,need_refinement=False,need_to_make_generic=False,chamber=None,term=None):
  input_output_creator = CreateInputsOutputs(inference_dir)
  # prev_actions.insert(0,"Miscellaneous")
  if need_refinement:
    prev_refined_actions=create_refinements(prev_actions)
  else:
    prev_refined_actions=prev_actions
  if need_to_make_generic:
    prev_generics = [make_generic(action,input_output_creator.get_generics()) for action in prev_refined_actions]
  else:
    prev_generics=prev_refined_actions
  prev_categories = [categorize(action) for action in prev_actions]
  input_vector = prev_input_vector or input_output_creator.create_input_vector(prev_generic="Start of bill.")
  # input_vector = prev_input_vector or input_output_creator.create_input_vector()
  for generic,categories in zip(prev_generics,prev_categories):
    input_vector = input_output_creator.create_input_vector(prev_input_vector=input_vector,prev_generic=generic,prev_categories=categories,chamber=chamber,term=term)
  input_vector = input_vector.astype(np.float32)
  sess = ort.InferenceSession(os.path.join(inference_dir,model_name+".onnx"))
  output=sess.run(None,{"input":input_vector.astype(np.float32)})[0]
  
  probabilities=input_output_creator.vector_to_probabilities(output)
  return probabilities

if __name__=="__main__":
  predicted_generics = predict_action_from_seq("model",
                                                                    prev_actions=["Introduced in the House.","Referred to [SUBCOMMITTEE]"],
                                                                    inference_dir="outputs/preprocess5/inference")
  # predicted_generics=[x for x in predicted_generics]
  
  display(sorted(predicted_generics.items(),key=lambda x:x[1],reverse=True))
  

# def predict_bill(bill_df,refine_first=True):
#   if refine_first:
#     refined_actions = refine_actions(bill_df["action"])
#   bill_df["generic"]=generic_refined_actions(refined_actions)
#   bill_df["categories"]=bill_df["action"].apply(categories_map)
#   bill_df.loc[-1]=([None for i in range(len(bill_df.columns))])
  
#   vecs = create_bill_vectors(bill_df)

#   state_dict = torch.load("./../outputs/models/08-04_withextras_bce_lr1e-5_beta1e-06/epoch200.pt")
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

# with open("./../outputs/data/test_data.csv","r") as file:
#   bill_dfs = [x[1] for x in pd.read_csv(file).groupby("bill_id")]
#   bill_dfs.sort(key=lambda x:len(x["action"]),reverse=True)

# predictions = predict_bill(bill_dfs[7000])
