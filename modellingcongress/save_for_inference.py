import torch
import argparse
import os
import onnx
from prepare_inputs_outputs import CreateInputsOutputs
import onnxruntime as ort
import numpy as np

parser = argparse.ArgumentParser(description="save model for inference")
parser.add_argument("-i","--inference_dir",default="outputs/preprocess5/inference",type=str, help="the directory for the data required for inference")  
parser.add_argument("-m","--model_path",type=str,default="outputs/preprocess5/models/lr3e-04_lassoweight1e-06_batch256/epoch100.pt")  
parser.add_argument("-o","--output_model_name",default="model",type=str, help="the output model name, defaults to last dir in model_path")  


args,unknown = parser.parse_known_args()

def load_model(model_path):
  state_dict = torch.load(model_path)
  # model=torch.nn.Linear(state_dict["model"]["weight"].shape[1],state_dict["model"]["weight"].shape[0])
  model=torch.nn.Sequential(torch.nn.Linear(state_dict["model"]["weight"].shape[1],state_dict["model"]["weight"].shape[0]),torch.nn.Softmax(dim=0))
  model[0].load_state_dict(state_dict["model"])
  # model.load_state_dict(state_dict["model"])
  return model


model = load_model(args.model_path)
output_model_name = args.output_model_name if args.output_model_name is not None else os.path.basename(os.path.dirname(args.model_path))
output_model_path=os.path.join(args.inference_dir,output_model_name+".onnx")
creator = CreateInputsOutputs(args.inference_dir)
input = creator.create_input_vector()
input1 = creator.create_input_vector(term=112)
torch.onnx.export(model,torch.tensor(input1).float(),output_model_path,input_names=["input"])
# onnx_model = onnx.load(output_model_path)
# sess = ort.InferenceSession(output_model_path)
