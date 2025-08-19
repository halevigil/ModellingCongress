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
import argparse


class ActionDataset(torch.utils.data.Dataset):
  def __init__(self,df):
    self.inputs = [np.concatenate([arr1, arr2,arr3,arr4,arr5,arr6]) for arr1, arr2, arr3, arr4,arr5,arr6 in zip(df['recent_generics'], df['cum_prev_generics'],df["recent_categories"],df["cum_prev_categories"],df["term"],df["chamber"])]
    self.outputs = [np.concatenate([arr1, arr2]) for arr1, arr2 in zip(df['output_generic'], df['output_categories'])]
    self.GENERICS_LEN = df.iloc[0]["output_generic"].shape[0]
    self.CATEGORIES_LEN = df.iloc[0]["output_categories"].shape[0]
  def input_len(self):
     return len(self.inputs[0])
  def output_len(self):
     return len(self.outputs[0])
  def generics_len(self):
     return self.GENERICS_LEN
  def categories_len(self):
     return self.CATEGORIES_LEN
  def __len__(self):
    return len(self.inputs)
  def __getitem__(self,idx):
    return self.inputs[idx],self.outputs[idx]


def train_model(preprocessing_dir,lr=3e-4,lasso_weight=1e-5,batch_size=256,continue_from=None,run_name=None,override_previous=False,end_epoch=300,n_epochs=None):
  hyperparams={"lr":lr,"lasso_weight":lasso_weight,"batch size":batch_size}
  print("hyperparameters:",hyperparams)
  ds = ActionDataset(pd.read_pickle(os.path.join(preprocessing_dir,"train_vectors.pkl")))
  train_dataset,val_dataset = torch.utils.data.random_split(ds,[0.8,0.2])

  train_loader = torch.utils.data.DataLoader(train_dataset,batch_size=batch_size,shuffle=True)
  val_loader = torch.utils.data.DataLoader(val_dataset,batch_size=None,shuffle=True)

  model = torch.nn.Linear(ds.input_len(),ds.output_len())
  optim = torch.optim.Adam(model.parameters(),lr=lr)
  folder = os.path.join(preprocessing_dir,run_name or "08-07_lr{:.0e}_lassoweight{:.0e}_batch{}".format(lr,lasso_weight,batch_size))
  log=""
  if not os.path.exists(folder):
    os.mkdir(folder)
    json.dump(hyperparams,open(os.path.join(folder,"hyperparameters.json"),"w"))
  
  prev_epochs = sorted([int(re.match(r"epoch([0-9]+)",file)[1]) for file in os.listdir(folder) if re.search(r"epoch[0-9]+",file)])
  if continue_from==-1:
    continue_from=prev_epochs[-1]
  if not prev_epochs and continue_from!=None and continue_from!=-1:
    raise Exception("Trying to continue a model that doesn't exist")
  if prev_epochs and continue_from==None and not override_previous:
    raise Exception("Trying to run a model that already exists without a continue_from or override_previous")
  if prev_epochs and continue_from and continue_from>prev_epochs[-1]:
    raise Exception("Trying to load from model state that does not exist")
  elif prev_epochs and continue_from and prev_epochs[-1]>continue_from and not override_previous:
    raise Exception("Trying to override runs without override_previous")
  if continue_from!=None:
    model.load_state_dict(torch.load(os.path.join(folder,f"epoch{continue_from}.pt"))["model"])
    optim.load_state_dict(torch.load(os.path.join(folder,f"epoch{continue_from}.pt"))["optim"])
  for epoch in prev_epochs:
    if continue_from and epoch>continue_from:
      os.remove(os.path.join(folder,f"epoch{continue_from}.pt"))
  pred_generics_loss_fn = torch.nn.CrossEntropyLoss()
  pred_categories_loss_fn = torch.nn.BCEWithLogitsLoss()
  if not os.path.isdir(folder):
      os.mkdir(folder)
  val_pred_generics_losses=[]
  start_epoch = continue_from+1 if continue_from else 0
  if n_epochs:
    end_epoch=start_epoch+n_epochs-1
  for epoch in range(start_epoch,end_epoch+1):
    for i,(inpt,output) in enumerate(train_loader):
      optim.zero_grad()
      inpt=inpt.float()
      output=output.float()
      pred = model(inpt)
      pred_generics_loss = pred_generics_loss_fn(pred[:,:ds.generics_len()],output[:,:ds.generics_len()])
      pred_categories_loss = pred_categories_loss_fn(pred[:,ds.generics_len():],output[:,ds.generics_len():])
      
      lasso_loss = torch.norm(model.weight,p=1)
      loss = pred_generics_loss+pred_categories_loss+lasso_weight*lasso_loss
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
        pred_generics_loss += pred_generics_loss_fn(pred[:ds.generics_len()],output[:ds.generics_len()])
        pred_categories_loss += pred_categories_loss_fn(pred[ds.generics_len():],output[ds.generics_len():])
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

  
  parser = argparse.ArgumentParser()
  parser.add_argument("-d","--preprocessing_dir",type=str,default="./outputs/preprocess0", help="the directory for this preprocessing run")
  parser.add_argument("--lr",type=float,default=3e-4,help="learning rate")
  parser.add_argument("--batch_size",type=int,default=256,help="batch size")
  parser.add_argument("--lasso_weight",type=float,default=1e-5,help="weight of lasso loss")
  parser.add_argument("--continue_from","-c",type=int,default=None,help="epoch to continue from (-1 for last model run)")
  parser.add_argument("--override_previous","-o",action="store_true",help="override previous model run")
  parser.add_argument("--end_epoch",type=int,default=300,help="end epoch")
  parser.add_argument("--n_epoch",type=int,default=None,help="number of epochs to run (overrides end epoch)")
  parser.add_argument("--run_name",type=str,default=None,help="name for model run (if none, defaults to hyperparameters)")
  args,unknown = parser.parse_known_args()

  for lr in [2e-4,3e-5,3e-6]:
    for lasso_weight in [1e-4,1e-5,1e-6]:
      for batch_size in [128,256,512]:
        try:
          train_model(preprocessing_dir=args.preprocessing_dir,lr=lr,batch_size=args.batch_size,
                  lasso_weight=args.lasso_weight,continue_from=args.continue_from, override_previous=False,end_epoch=args.end_epoch,n_epochs=args.n_epoch)
        except:
          continue
  # train_model(preprocessing_dir=args.preprocessing_dir,lr=args.lr,batch_size=args.batch_size,
  #             lasso_weight=args.lasso_weight,continue_from=args.continue_from, override_previous=True,end_epoch=args.end_epoch,n_epochs=args.n_epoch)