import json
from collections import defaultdict
if __package__:
  from .make_generics import edit_distance_below
  from .clustering_util import cluster
else:
  from make_generics import edit_distance_below
  from clustering_util import cluster

import argparse
import os

parser = argparse.ArgumentParser(description="makes generics after llm refinement by combining actions with small edit distance")

parser.add_argument("-d","--preprocessing_dir",type=str,default="/Users/gilhalevi/Library/CloudStorage/OneDrive-Personal/Code/ModellingCongress/outputs/preprocess0", help="the directory for this preprocessing run")
parser.add_argument("--threshold",type=float,default=1/7,help="the max value of threshold*max(action1 length,action 2 length) for which the two actions will have the same generic")
args,unknown = parser.parse_known_args()

with open(os.path.join(args.preprocessing_dir,"generics_dict_manual.json"),"r") as file:
  generics_dict_manual = json.load(file)
with open(os.path.join(args.preprocessing_dir,"refinement_map.json"),"r") as file:
  refinement_map=json.load(file)
# display(refinement_map)

generics_dict_manual_llm=defaultdict(list)
unrefined=[]
for name in generics_dict_manual:
  if name in refinement_map:
    generics_dict_manual_llm[refinement_map[name]].extend(generics_dict_manual[name])
  else:
    unrefined.append(name)
    generics_dict_manual_llm[name].extend(generics_dict_manual[name])

# display(generics_dict_manual_llm.keys())
generics_manual_llm = list(generics_dict_manual_llm.keys())


generics_manual_llm_clusters = cluster(generics_manual_llm, lambda x,y:edit_distance_below(x,y,1/7*max(len(x),len(y))))
generics_dict_manual_llm_manual=defaultdict(list)
for name in generics_manual_llm_clusters:
  for name2 in generics_manual_llm_clusters[name]:
    generics_dict_manual_llm_manual[name].extend(generics_dict_manual_llm[name2])

with open(os.path.join(args.preprocessing_dir,"generics_dict_manual_llm_manual.json"),"w") as file:
  json.dump(generics_dict_manual_llm_manual,file,indent=2)