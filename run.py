import argparse
import os
parser = argparse.ArgumentParser()
parser.add_argument("-d","--preprocessing_dir",type=str,default="outputs/preprocess2", help="the directory for this preprocessing run")
  
args,unknown = parser.parse_known_args()

if not os.path.exists(args.preprocessing_dir):
  os.mkdir(args.preprocessing_dir)

import subprocess

subprocess.run(["python","modellingcongress/prepare_initial_data.py","-d",str(args.preprocessing_dir)])
subprocess.run(["python","modellingcongress/make_generics.py","-d",str(args.preprocessing_dir)])
subprocess.run(["python","modellingcongress/llm_refinement.py","-d",str(args.preprocessing_dir)])
subprocess.run(["python","modellingcongress/make_generics_post_llm.py","-d",str(args.preprocessing_dir)])
subprocess.run(["python","modellingcongress/prepare_inputs_outputs.py","-d",str(args.preprocessing_dir)])
subprocess.run(["python","modellingcongress/prediction_model.py","-d",str(args.preprocessing_dir)])
model_run = list(os.listdir(os.path.join(args.preprocessing_dir,"models")))[0]
model = list(os.listdir(os.path.join(args.preprocessing_dir,"models",model_run)))[-2]
subprocess.run(["python","modellingcongress/save_for_inference.py","-m",str(os.path.join(args.preprocessing_dir,"models",model_run,model),"-i",str(os.path.join(args.processing_dir,"inference")))])

