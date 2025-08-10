# buckets_refined_combined=bucket_names_refined_combined
import json
from collections import defaultdict
import bucketing_fn
buckets_refined_combined=defaultdict(list)

with open("outputs/buckets_manual.json","r") as file:
  buckets_manual = json.load(file)
with open("outputs/llm_refinement_map.json","r") as file:
  refinement_map=json.load(file)

buckets_manual_llm=defaultdict(list)
for name in buckets_manual:
  refinement_map[buckets_manual_llm[name]].extend(buckets_manual[name])

with open("outputs/buckets_manual_llm.json","w") as file:
  json.dump(buckets_manual_llm,file)
with open("outputs/buckets_manual_llm.json","r") as file:
  buckets_manual_llm=json.load(file)

buckets_manual_llm_names = list(buckets_manual_llm.keys())
bucket_names_buckets = bucketing_fn.bucket(buckets_manual_llm_names)


buckets_manual_llm_manual=defaultdict(list)
for name in bucket_names_buckets:
  for name2 in bucket_names_buckets[name]:
    buckets_manual_llm_manual[name].extend(buckets_manual_llm[name2])

with open("outputs/buckets_manual_llm_manual.json","w") as file:
  json.dump(buckets_manual_llm_manual,file)