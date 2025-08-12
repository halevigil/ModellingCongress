import json
import pandas as pd
from collections import defaultdict, Counter
import sklearn
import sklearn.model_selection
import random
import os

if not os.path.isdir("../outputs/data"):
  os.mkdir("../outputs/data")
if not os.path.isdir("../outputs/data"):
  os.mkdir("../outputs/data")

datasets = [f"../data/{term*2+2009-222}-{term*2+2010-222}_{term}th_Congress/csv/history.csv" for term in range(111,120)]
history_df_by_term=[pd.read_csv(ds) for ds in datasets]
for i,df in enumerate(history_df_by_term):
  df["term"]=i+111
history_df=pd.concat(history_df_by_term)


bills=[group for (bill_id,group) in history_df.groupby("bill_id")]
for bill in bills:
  bill.loc[-1]=bill.iloc[-1]
  bill.at[-1,"action"]="Last Action"

history_df.to_csv("../outputs/data/data_pregenericing.csv",index=False)