import json
import dotenv
import openai
import os
import time
dotenv.load_dotenv()
client =openai.Client(api_key=os.environ["OPENAI_API_KEY"])

with open("outputs/buckets_08-04_manual-llm-manual.json","r") as file:
  buckets = json.load(file)

with open("outputs/embedding/08-07/input.jsonl","w") as file:
  for i,name in enumerate(buckets):
    json.dump({"custom_id":f"buckets{i}","method":"POST","url":"/v1/embeddings",
               "body":{"model":"text-embedding-3-small","input":name}},file)
    file.write("\n")

file = client.files.create(file=open("outputs/embedding/08-07/input.jsonl","rb"),purpose="batch")
batch = client.batches.create(endpoint="/v1/embeddings",input_file_id=file.id,completion_window="24h")
display(client.batches.retrieve(batch_id=batch.id))
with open("outputs/embedding/08-07/output.jsonl","w") as file:
  file.write(client.files.content(file_id="file-Y2zRpjFiMjYABhrBeKftQs").text)