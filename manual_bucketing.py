import pandas as pd
import re
import json
import numpy as np
import bucketing_fn

# Extract all committee spellings

bills_datasets = [f"data/US/{term*2+2009-222}-{term*2+2010-222}_{term}th_Congress/csv/bills.csv" for term in range(111,120)]
bills_df=pd.concat([pd.read_csv(dataset) for dataset in bills_datasets])
committees=[committee for committee in set(bills_df["committee"]) if type(committee)==str and committee !=" " and committee!=""]
committee_search_item={}
def all_committee_spellings(committees):
  committee_spellings={committee:[committee] for committee in committees}
  for committee in committees:
    committee_spellings[committee]=[]
    if not re.search("committee",committee.lower()):
      committee_spellings[committee].append(committee + " Committee")
      committee_spellings[committee].append("Committee on "+committee)
      committee_spellings[committee].append("Committee on the "+committee)
      if re.search("Senate",committee):
        committee_spellings[committee].append(re.sub("Senate ","",committee)+ " Committee")
        committee_spellings[committee].append("Senate Committee on the "+re.sub("Senate ","",committee))
        committee_spellings[committee].append("Senate Committee on "+re.sub("House ","",committee))
        committee_spellings[committee].append("Committee on the "+re.sub("Senate ","",committee))
        committee_spellings[committee].append("Committee on "+re.sub("Senate ","",committee))
      if re.search("House",committee):
        committee_spellings[committee].append(re.sub("House ","",committee)+ " Committee")
        committee_spellings[committee].append("House Committee on the "+re.sub("House ","",committee))
        committee_spellings[committee].append("House Committee on "+re.sub("House ","",committee))
        committee_spellings[committee].append("Committee on the "+re.sub("House ","",committee))
        committee_spellings[committee].append("Committee on "+re.sub("House ","",committee))
    committee_spellings[committee].append(committee)
    if re.search("(House )|(Senate )",committee):
      committee_spellings[committee].append(re.sub("(House )|(Senate )","",committee))
    if re.search("committee",committee):
        committee_spellings[committee].append(re.sub("(Subcommittee on )|(Subcommittee for )|(Subcommittee )|( Subcommittee)|(Committee on )|(Committee for )|( Committee)|(Committee )","",committee))
    if re.search("committee",committee) and re.search("(House )|(Senate )",committee):
        committee_spellings[committee].append(re.sub("(Subcommittee on )|(Subcommittee for )|(Subcommittee )|( Subcommittee)|(Committee on )|(Committee for )|( Committee)|(Committee )|(Senate )|(House )","",committee))
    committee_search_item[committee]=re.sub("(Subcommittee on )|(Subcommittee for )|(Subcommittee )|( Subcommittee)|(Committee on )|(Committee for )|( Committee)|(Committee )|(Senate )|(House )","",committee)
    return committee_spellings,committee_search_item
def get_action_committee_map(actions,committee_search_item):
  action_committees_map={}
  for action in actions:
    action_committees_map[action]=[]
    for committee in committees:
      if re.search(committee_search_item[committee],action):
        action_committees_map[action].append(committee)

# Process action to get rid of committee names, representative names, times, parentheticals, numbers
# in order to compare actions. These are bill specific and we want the generic version
action_process_map={} # a cache for the below function
def process_action(action):
  if action in action_process_map:
    return action_process_map[action]
  action=action.lower()
  action=re.sub("committees","committee",action)
  committees=[committee for committee in committee_search_item if re.search(committee_search_item[committee].lower(),action) and not re.search("subcommittee",action.lower())]
  subcommittees=[committee for committee in committee_search_item if re.search(committee_search_item[committee].lower(),action) and re.search("subcommittee",action.lower())]
  if len(committees)>0:
    committee_spellings_regex="|".join(["("+")|(".join(committee_spellings[committee])+")" for committee in committees]).lower()
    action = re.sub(committee_spellings_regex,"committee",action)
  if len(subcommittees)>0:
    committee_spellings_regex="|".join(["("+")|(".join(committee_spellings[committee])+")" for committee in subcommittees]).lower()
    action = re.sub(committee_spellings_regex,"subcommittee",action)
  action=re.sub(r"\,|\.|\-"," ",action)
  action=re.sub(r"[0-9]","",action)
  action=re.sub(r" +"," ",action)
  action = re.sub(r'\b(mr|mrs|ms|senator|representative) \w+\s', 'representative ',action)
  action = re.sub(r'(\w+ hour)|(\w+ minutes)', 'time',action)
  action = re.sub(r'\(.*?\)', '',action)
  action=re.sub(r" +"," ",action)
  action_process_map[action]=action
  return action


# Compute edit distance between two lists/strings
def edit_distance(l1,l2):
  dp=np.empty((len(l2)+1,len(l1)+1))
  dp[0]=range(len(l1)+1)
  dp[:,0]=range(len(l2)+1)
  for i in range(1,len(l2)+1):
    for j in range(1,len(l1)+1):
      dp[i,j]=min(dp[i][j-1]+1,dp[i-1][j]+1,dp[i-1][j-1] if l1[j-1]==l2[i-1] else dp[i-1][j-1]+1)
  return dp[-1,-1]

# Check if the edit distance between two lists/strings is below a threshold
# Uses tricks to speed up operation
def edit_distance_below(l1,l2,threshold):
  if abs(len(l1)-len(l2))>threshold:
    return False
  dp=np.empty((len(l2)+1,len(l1)+1))
  dp[0]=range(len(l1)+1)
  dp[:,0]=range(len(l2)+1)
  min_dist=1
  max_num=max(len(l2),len(l1))+1
  # loop in diagonal order, like this: 
  #  1 2 4
  #  3 5 7
  #  6 8 9
  for i in range(2*len(l2)+2):
    min_dist_last=min_dist
    min_dist=max_num
    for j in range(max(i-len(l1)-1,0),min(i+1,len(l2)+1)):
      col=i-j
      row=j
      if row > len(l2) or col > len(l1):
        continue
      if row>0 and col>0:
        dp[row,col]=min(dp[row][col-1]+1,dp[row-1][col]+1,dp[row-1][col-1] if l2[row-1]==l1[col-1] else dp[row-1][col-1]+1)
      if dp[row,col]<min_dist:
        min_dist=dp[row,col]
    #  quit early if two consecutive diagonals have no values below threshold 
    if min_dist<max_num and min(min_dist,min_dist_last)>threshold:
      return False
  return dp[-1,-1]<=threshold


# check if two actions are similar (have small edit distance) after processing
def similar_after_processing(action1,action2,threshold):
  action1=process_action(action1)
  action2=process_action(action2)

  mustmatch_regexes=["subcommittee"]
  for regex in mustmatch_regexes:
    if not (bool(re.search(regex,action1))==bool(re.search(regex,action2))):
      return False
  return edit_distance_below(action1,action2,threshold*max(len(action1),len(action2)))

def special_buckets_f(action):
  if re.search(r"^h\.amdt\.[0-9]*? amendment \(.*?\) in the nature of a substitute offered by",action.lower()):
    return "H.Amdt. in the nature of a substitute offered by"
  if re.search(r"^h\.amdt\.[0-9]*? amendment \(.*?\) offered by",action.lower()):
    return "H.Amdt. offered by"
  if re.search(r"^s\.amdt\.[0-9]*? amendment sa [0-9]*? proposed by",action.lower()):
    return "S.Amdt. offered by"
  if re.search(r"the.*? house.*? proceeded.*? with.*? of debate.*? on.*? the.*? amendment",action.lower()):
    return "DEBATE - The House proceeded with 10 minutes of debate on the Broun (GA) motion to recommit with instructions"
  return None

if __name__=="__main__":
  committee_spellings,committee_search_item=all_committee_spellings(committees)
  # Map each action to the committees it is in
  datasets = [f"data/US/{term*2+2009-222}-{term*2+2010-222}_{term}th_Congress/csv/history.csv" for term in range(111,120)]
  history_df=pd.concat([pd.read_csv(dataset) for dataset in datasets])
  actions = list(history_df["action"])

  # returns a map from action to the committee it is in
  action_committees_map=get_action_committee_map(actions,committee_search_item)
  json.dump(action_committees_map,open("./outputs/action_committees_map.json","w"),indent=2)
  action_committees_map = json.load(open("./outputs/action_committees_map.json","r"))


  buckets = bucketing_fn.bucket(actions,similar_after_processing,special_buckets_f)
  with open("outputs/buckets.json","w") as file:
    json.dump(buckets,file)







