import json
import pandas as pd
from collections import defaultdict, Counter
import numpy as np
import sklearn
import sklearn.linear_model
import sklearn.preprocessing
import torch
import os
import shutil
import builtins
import sys
import re
import sklearn

with open("outputs/common_bucket_names.json","r") as file:
  common_bucket_names = json.load(file)
with open("outputs/common_extra_bucket_names.json","r") as file:
  common_extra_bucket_names = json.load(file)
common_bucket_names_inv = {name:i for i,name in enumerate(common_bucket_names)}
common_extra_bucket_names_inv = {name:i for i,name in enumerate(common_bucket_names)}

MIN_TERM=111
N_TERMS = 9
alpha=2/3
def create_bill_vectors_unscaled(bill_df):
  out=[]
  predecessors_buckets=np.zeros(len(common_bucket_names))
  predecessor_buckets=np.zeros(len(common_bucket_names))
  predecessors_extra_buckets=np.zeros(len(common_extra_bucket_names))
  predecessor_extra_buckets=np.zeros(len(common_extra_bucket_names))
  curr_bucket = np.zeros(len(common_bucket_names))
  curr_extra_buckets = np.zeros(len(common_extra_bucket_names))
  for i,row in bill_df.iterrows():
    prev_bucket=np.array(curr_bucket)
    prev_extra_buckets=np.array(curr_extra_buckets)
    curr_bucket = np.zeros(len(common_bucket_names))
    curr_extra_buckets = np.zeros(len(common_extra_bucket_names))
    output_bucket = np.zeros(len(common_bucket_names)+1)
    if row["bucket"] in common_bucket_names_inv:
      curr_bucket[common_bucket_names_inv[row.bucket]]=1
      output_bucket[common_bucket_names_inv[row.bucket]]=1
    else:
      output_bucket[-1]=1
    if row["extra_buckets"]:
      for extra_bucket in row["extra_buckets"]: 
        if extra_bucket in common_extra_bucket_names_inv:
          curr_extra_buckets[common_extra_bucket_names_inv[extra_bucket]]=1
    
    predecessor_buckets=(1-alpha)*np.array(predecessor_buckets)+alpha*np.array(prev_bucket)
    predecessors_buckets=np.array(predecessors_buckets)+prev_bucket
  

    predecessor_extra_buckets=(1-alpha)*predecessor_extra_buckets+alpha*np.array(prev_extra_buckets)
    predecessors_extra_buckets=np.array(predecessors_extra_buckets)+prev_extra_buckets
    chamber = np.zeros(2)
    if row["chamber"]=="House":
      chamber[0]=1
    elif row["chamber"]=="Senate":
      chamber[1]=1
    term=np.zeros(N_TERMS)
    if row["term"]:
      term[row["term"]-MIN_TERM]=1
    entry={"predecessor_buckets":predecessor_buckets,"predecessors_buckets":predecessors_buckets,"predecessor_extra_buckets":predecessor_extra_buckets,"predecessors_extra_buckets":predecessors_extra_buckets,"term":term,"chamber":chamber,
          "output_bucket":output_bucket,"output_extra_buckets":curr_extra_buckets} # print(entry)
    out.append(entry)
  return out

def concat(ls):
  out=[]
  for l in ls:
    out+=l
  return out


bills = pd.read_csv("outputs/data/train_data.csv").groupby("bill_id")
data = pd.DataFrame(concat(create_bill_vectors_unscaled(bill) for i,bill in bills))
preds_buckets = np.stack(data["predecessors_buckets"],axis=0)
preds_extra_buckets = np.stack(data["predecessors_extra_buckets"],axis=0)
scaler_preds_buckets = sklearn.preprocessing.StandardScaler().fit(preds_buckets)
scaler_preds_extra_buckets = sklearn.preprocessing.StandardScaler().fit(preds_extra_buckets)
def create_bill_vectors(bill):
  out = create_bill_vectors_unscaled(bill)
  for d in out:
    d["predecessors_buckets"] = scaler_preds_buckets.transform([d["predecessors_buckets"]])[0]
    d["predecessors_extra_buckets"] = scaler_preds_extra_buckets.transform([d["predecessors_extra_buckets"]])[0]
  return out
preds_buckets_scaled=scaler_preds_buckets.transform(preds_buckets)
preds_extra_buckets_scaled=scaler_preds_extra_buckets.transform(preds_extra_buckets)


data["predecessors_buckets"]=list(preds_buckets_scaled)
data["predecessors_extra_buckets"]=list(preds_extra_buckets_scaled)
data_dict = [dict(entry[1]) for entry in data.iterrows()]
np.savez_compressed("outputs/prediction_vecs_08-07.npz",data_dict)
np.load("outputs/prediction_vecs_08-07.npz",allow_pickle=True)["arr_0"][0]



class ActionDataset(torch.utils.data.Dataset):
  def __init__(self,data):
    self.inputs = [np.concatenate([entry["predecessor_buckets"],entry["predecessors_buckets"],entry["predecessor_extra_buckets"],entry["predecessors_extra_buckets"],entry["term"],entry["chamber"]]) for entry in data]
    # self.inputs = sklearn.preprocessing.StandardScaler().fit_transform(self.inputs)
    self.outputs = [np.concatenate((entry["output_bucket"],entry["output_extra_buckets"])) for entry in data]
  def __len__(self):
    return len(self.inputs)
  def __getitem__(self,idx):
    return self.inputs[idx],self.outputs[idx]

ds = ActionDataset(np.load("outputs/prediction_vecs_08-07.npz",allow_pickle=True)["arr_0"])
train_dataset,val_dataset = torch.utils.data.random_split(ds,[0.8,0.2])


def train_model(lr=1e-4,lasso_weight=1e-7,batch_size=32,extra_pred_weight=1,continue_from=None,special_name="",override_previous=False,end_epoch=301,n_epochs=None):
  hyperparams={"lr":lr,"lasso_weight":lasso_weight,"batch size":batch_size,"extra_pred_weight":extra_pred_weight}
  print("hyperparameters:",hyperparams)
  train_loader = torch.utils.data.DataLoader(train_dataset,batch_size=batch_size,shuffle=True)
  val_loader = torch.utils.data.DataLoader(val_dataset,batch_size=None,shuffle=True)

  model = torch.nn.Linear(len(common_bucket_names)*2+len(common_extra_bucket_names)*2+N_TERMS+2,len(common_bucket_names)+1+len(common_extra_bucket_names))
  optim = torch.optim.Adam(model.parameters(),lr=lr)
  folder = "outputs/models/08-07_lr{:.0e}_lassoweight{:.0e}_batch{}_extra{}{}".format(lr,lasso_weight,batch_size,extra_pred_weight,special_name)
  log=""
  if not os.path.exists(folder):
    os.mkdir(folder)
    json.dump(hyperparams,open(os.path.join(folder,"hyperparameters.json"),"w"))
  
  prev_epochs = sorted([int(re.match(r"epoch([0-9]+)",file)[1]) for file in os.listdir(folder) if re.search(r"epoch[0-9]+",file)])
  continue_from=prev_epochs[-1] if continue_from==-1 and prev_epochs else continue_from
  if not prev_epochs and continue_from!=None and continue_from!=-1:
    raise Exception("Trying to continue a model that doesn't exist")
  if prev_epochs and continue_from==None :
    raise Exception("Trying to run a model that already exists without a continue_from")
  if prev_epochs and  continue_from>prev_epochs[-1]:
    raise Exception("Trying to load from model state that does not exist")
  elif prev_epochs and prev_epochs[-1]>continue_from and not override_previous:
    raise Exception("Trying to override runs without override_previous")
  if continue_from==-1:
    continue_from=prev_epochs[-1]
  model.load_state_dict(torch.load(os.path.join(folder,f"epoch{continue_from}.pt"))["model"])
  optim.load_state_dict(torch.load(os.path.join(folder,f"epoch{continue_from}.pt"))["optim"])
  for epoch in prev_epochs:
    if epoch>continue_from:
      os.remove(os.path.join(folder,f"epoch{continue_from}.pt"))
  pred_buckets_loss_fn = torch.nn.CrossEntropyLoss()
  pred_extra_buckets_loss_fn = torch.nn.BCEWithLogitsLoss()
  if not os.path.isdir(folder):
      os.mkdir(folder)
  val_pred_buckets_losses=[]
  start_epoch = continue_from+1 if continue_from else 0
  if n_epochs:
    end_epoch=start_epoch+n_epochs
  for epoch in range(start_epoch,end_epoch):
      for i,(inpt,output) in enumerate(train_loader):
          optim.zero_grad()
          inpt=inpt.float()
          output=output.float()
          pred = model(inpt)
          pred_buckets_loss = pred_buckets_loss_fn(pred[:,:len(common_bucket_names)+1],output[:,:len(common_bucket_names)+1])
          pred_extra_buckets_loss = pred_extra_buckets_loss_fn(pred[:,len(common_bucket_names)+1:],output[:,len(common_bucket_names)+1:])
          
          lasso_loss = torch.norm(model.weight,p=1)
          loss = pred_buckets_loss+extra_pred_weight*pred_extra_buckets_loss+lasso_weight*lasso_loss
          loss.backward()
          optim.step()
      with torch.no_grad():
          pred_buckets_loss=torch.scalar_tensor(0)
          pred_extra_buckets_loss=torch.scalar_tensor(0)
          lasso_loss=torch.norm(model.weight,p=1)
          for inpt,output in val_loader:
              inpt=inpt.float()
              output=output.float()
              pred = model(inpt)
              pred_buckets_loss += pred_buckets_loss_fn(pred[:len(common_bucket_names)+1],output[:len(common_bucket_names)+1])
              pred_extra_buckets_loss += pred_extra_buckets_loss_fn(pred[len(common_bucket_names)+1:],output[len(common_bucket_names)+1:])
          pred_buckets_loss/=len(val_loader)
          pred_extra_buckets_loss/=pred_extra_buckets_loss
      val_pred_buckets_losses.append(pred_buckets_loss)
      
      loss_str=f"epoch {epoch}. lasso loss:{lasso_loss} log pred buckets loss:{float(torch.log10(pred_buckets_loss))},log pred extra buckets loss:{float(torch.log10(pred_extra_buckets_loss))} pred buckets loss:{float(pred_buckets_loss)} pred extra buckets loss:{float(pred_extra_buckets_loss)}"
      log+=loss_str+"\n"
      if epoch%5==0:
          torch.save({"model":model.state_dict(),"optim":optim.state_dict()},folder+f"/epoch{epoch}.pt")
          with open(os.path.join(folder,"log"),"w") as file:
            file.write(log)
      if len(val_pred_buckets_losses)>10 and val_pred_buckets_losses[-1]+val_pred_buckets_losses[-2]>val_pred_buckets_losses[-3]+val_pred_buckets_losses[-4]>val_pred_buckets_losses[-5]+val_pred_buckets_losses[-6]:
        break
      print(loss_str)
if __name__=="__main__":
  # train_model(lr=3e-4,lasso_weight=1e-5,batch_size=256,load_epoch=100)
  for lr in [1e-3,1e-4,1e-5]:
    for lasso_weight in [1e-3,1e-4,1e-5]:
      for batch_size in [256]:
        # try:
        train_model(lr=lr,lasso_weight=lasso_weight,batch_size=batch_size,continue_from=-1)
