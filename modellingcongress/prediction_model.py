# Trains a prediction market to forecast future actions based on previous actions

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


class ActionDataset(torch.utils.data.Dataset):
  def __init__(self,df):
    self.inputs = [np.concatenate([arr1, arr2,arr3,arr4]) for arr1, arr2, arr3, arr4 in zip(df['recent_generics'], df['cum_prev_generics'],df["recent_categories"],df["cum_prev_categories"],df["term"],df["chamber"])]
    self.outputs = [np.concatenate([arr1, arr2]) for arr1, arr2 in zip(df['output_generic'], df['output_categories'])]
    self.REGULAR_genericS_LEN = len(df.at([0,"recent_generics"]))
    self.categories_LEN = len(df.at([0,"recent_categories"]))
  def input_len(self):
     return len(self.inputs[0])
  def output_len(self):
     return len(self.outputs[0])
  def regular_generics_len(self):
     return self.REGULAR_genericS_LEN
  def categories_len(self):
     return self.categories_LEN
  def __len__(self):
    return len(self.inputs)
  def __getitem__(self,idx):
    return self.inputs[idx],self.outputs[idx]

ds = ActionDataset(pd.read_pickle("./outputs/prediction_vecs"))
train_dataset,val_dataset = torch.utils.data.random_split(ds,[0.8,0.2])


def train_model(lr=3e-4,lasso_weight=1e-5,batch_size=256,extra_pred_weight=1,continue_from=None,special_name="",override_previous=False,end_epoch=301,n_epochs=None):
  hyperparams={"lr":lr,"lasso_weight":lasso_weight,"batch size":batch_size,"extra_pred_weight":extra_pred_weight}
  print("hyperparameters:",hyperparams)
  train_loader = torch.utils.data.DataLoader(train_dataset,batch_size=batch_size,shuffle=True)
  val_loader = torch.utils.data.DataLoader(val_dataset,batch_size=None,shuffle=True)

  model = torch.nn.Linear(ds.input_len,ds.output_len)
  optim = torch.optim.Adam(model.parameters(),lr=lr)
  folder = "./outputs/models/08-07_lr{:.0e}_lassoweight{:.0e}_batch{}_extra{}{}".format(lr,lasso_weight,batch_size,extra_pred_weight,special_name)
  log=""
  if not os.path.exists(folder):
    os.mkdir(folder)
    json.dump(hyperparams,open(os.path.join(folder,"hyperparameters.json"),"w"))
  
  prev_epochs = sorted([int(re.match(r"epoch([0-9]+)",file)[1]) for file in os.listdir(folder) if re.search(r"epoch[0-9]+",file)])
  continue_from=prev_epochs[-1] if continue_from==-1 and prev_epochs else continue_from
  if not prev_epochs and continue_from!=None and continue_from!=-1:
    raise Exception("Trying to continue a model that doesn't exist")
  if prev_epochs and continue_from==None:
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
  pred_generics_loss_fn = torch.nn.CrossEntropyLoss()
  pred_categories_loss_fn = torch.nn.BCEWithLogitsLoss()
  if not os.path.isdir(folder):
      os.mkdir(folder)
  val_pred_generics_losses=[]
  start_epoch = continue_from+1 if continue_from else 0
  if n_epochs:
    end_epoch=start_epoch+n_epochs
  for epoch in range(start_epoch,end_epoch):
    for i,(inpt,output) in enumerate(train_loader):
      optim.zero_grad()
      inpt=inpt.float()
      output=output.float()
      pred = model(inpt)
      pred_generics_loss = pred_generics_loss_fn(pred[:,:ds.regular_generics_len()])
      pred_categories_loss = pred_categories_loss_fn(pred[:,ds.regular_generics_len():])
      
      lasso_loss = torch.norm(model.weight,p=1)
      loss = pred_generics_loss+extra_pred_weight*pred_categories_loss+lasso_weight*lasso_loss
      loss.backward()
      optim.step()
    with torch.no_grad():
      pred_generics_loss=torch.scalar_tensor(0)
      pred_categories_loss=torch.scalar_tensor(0)
      lasso_loss=torch.norm(model.weight,p=1)
      for inpt,output in val_loader:
        inpt=inpt.float()
        output=output.float()
        pred = model(inpt)
        pred_generics_loss += pred_generics_loss_fn(pred[:ds.regular_generics_len()],output[:ds.regular_generics_len()])
        pred_categories_loss += pred_categories_loss_fn(pred[ds.regular_generics_len():],output[ds.regular_generics_len():])
      pred_generics_loss/=len(val_loader)
      pred_categories_loss/=pred_categories_loss
    val_pred_generics_losses.append(pred_generics_loss)
    
    loss_str=f"epoch {epoch}. lasso loss:{lasso_loss} log pred generics loss:{float(torch.log10(pred_generics_loss))},log pred extra generics loss:{float(torch.log10(pred_categories_loss))} pred generics loss:{float(pred_generics_loss)} pred extra generics loss:{float(pred_categories_loss)}"
    log+=loss_str+"\n"
    if epoch%5==0:
        torch.save({"model":model.state_dict(),"optim":optim.state_dict()},folder+f"/epoch{epoch}.pt")
        with open(os.path.join(folder,"log"),"w") as file:
          file.write(log)
    if len(val_pred_generics_losses)>10 and val_pred_generics_losses[-1]+val_pred_generics_losses[-2]>val_pred_generics_losses[-3]+val_pred_generics_losses[-4]>val_pred_generics_losses[-5]+val_pred_generics_losses[-6]:
      break
    print(loss_str)
if __name__=="__main__":
  train_model()