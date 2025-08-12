# generics_refined_combined=generic_names_refined_combined
import json
from collections import defaultdict
import modellingcongress.genericing_util as genericing_util
generics_refined_combined=defaultdict(list)

with open("../outputs/generics_manual.json","r") as file:
  generics_manual = json.load(file)
with open("../outputs/llm_refinement_map.json","r") as file:
  refinement_map=json.load(file)

generics_manual_llm=defaultdict(list)
for name in generics_manual:
  refinement_map[generics_manual_llm[name]].extend(generics_manual[name])

with open("../outputs/generics_manual_llm.json","w") as file:
  json.dump(generics_manual_llm,file)
with open("../outputs/generics_manual_llm.json","r") as file:
  generics_manual_llm=json.load(file)

generics_manual_llm_names = list(generics_manual_llm.keys())
generic_names_generics = genericing_util.generic(generics_manual_llm_names)


generics_manual_llm_manual=defaultdict(list)
for name in generic_names_generics:
  for name2 in generic_names_generics[name]:
    generics_manual_llm_manual[name].extend(generics_manual_llm[name2])

with open("../outputs/generics_manual_llm_manual.json","w") as file:
  json.dump(generics_manual_llm_manual,file)