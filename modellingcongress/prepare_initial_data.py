import json
import pandas as pd
from collections import defaultdict, Counter
import sklearn
import sklearn.model_selection
import random
import os
import argparse


parser = argparse.ArgumentParser(description="prepare data before creating generics")
parser.add_argument("-d","--preprocessing_dir",type=str,default="outputs/preprocess2", help="the directory for this preprocessing run")
args,unknown = parser.parse_known_args()

datasets = [f"./data/{term*2+2009-222}-{term*2+2010-222}_{term}th_Congress/csv/history.csv" for term in range(111,120)]
history_df_by_term=[pd.read_csv(ds) for ds in datasets]
for i,df in enumerate(history_df_by_term):
  df["term"]=i+111
history_df=pd.concat(history_df_by_term)


bills=[group for (bill_id,group) in history_df.groupby("bill_id")]

for i,bill in enumerate(bills):
  first_row = bill.iloc[0].copy()
  first_row["action"]= "Start of bill."
  last_row = bill.iloc[-1].copy()
  last_row["action"] = "No further actions."
  bills[i]=pd.concat([pd.DataFrame([first_row]),bill,pd.DataFrame([last_row])])
history_df_with_end=pd.concat(bills)
history_df_with_end.to_csv(os.path.join(args.preprocessing_dir,"initial_data.csv"),index=False)