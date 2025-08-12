import json
import pandas as pd
from collections import defaultdict, Counter
import sklearn
import sklearn.model_selection
import random
import os
import argparse


parser = argparse.ArgumentParser(description="prepare data before creating generics")
parser.add_argument("-d","--preprocessing_dir",type=str,default="../outputs/preprocess0.json")
args,unknown = parser.parse_known_args()

datasets = [f"../data/{term*2+2009-222}-{term*2+2010-222}_{term}th_Congress/csv/history.csv" for term in range(111,120)]
history_df_by_term=[pd.read_csv(ds) for ds in datasets]
for i,df in enumerate(history_df_by_term):
  df["term"]=i+111
history_df=pd.concat(history_df_by_term)


bills=[group for (bill_id,group) in history_df.groupby("bill_id")]
for bill in bills:
  bill.loc[-1]=bill.iloc[-1]
  bill.at[-1,"action"]="Last Action"

history_df.to_csv(os.path.join(args.d,"data_no_generics"),index=False)