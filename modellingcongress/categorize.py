# Every action belongs to exactly 1 generic
# These categories are extra classifications that 
# subsume generics and are not mutually exclusive

from collections import defaultdict
import re
import pandas as pd
import json
import argparse
import os


# maps an action to the categories it belongs to
regexes={r"^All points of consideration against consideration.*? are waived",\
         r"^ANNOUNCEMENT",r"^APPOINTMENT OF CONFEREE",r"^APPOINTMENT OF.*? CONFEREE",\
          r"^forwarded by subcommittee to full committee",r"^GENERAL DEBATE",r"^Introduced in the Senate",\
          r"^MOMENT OF SILENCE",r"^Motion by \[SENATOR\] to (commit|refer) to",r"^Motion (by \[SENATOR\] )?to concur",\
          r"^Motion by \[SENATOR\] to instruct",r"^Motion by \[SENATOR\] to reconsider",r"^Motion to disagree",\
          r"^Motion to discharge",r"^Motion to proceed to consideration",r"^Motion to table the motion",
          r"^Motion to waive",r"^NOTIFICATION OF INTENT","^ORDER OF BUSINESS","^ORDER OF PROCEDURE",
          r"^On agreeing to the resolution",r"On agreeing to the \[REPRESENTATIVE\] amendment",\
          r"On motion that the House (dis)?agree to the Senate amendment",\
          r"On motion that the House agree with an amendment",\
          r"On motion that the House (dis)?agree",
          r"On motion that the House instruct conferees",
          r"^On motion (that the House |to )suspend the rules",r"^On passage",r"^On ordering the previous question",
          r"^On questioning of consideration",r"^Ordered to be [r|R]eported",r"^point of order",
          r"^POSTPONED CONSIDERATION OF VETO MESSAGE",r"^POSTPONED PROCEEDING",r"POSTPONED ROLL CALL VOTE",
          r"^Passed Senate",r"^Previous question shall be considered as ordered",\
          r"Previous question shall be considered as ordered",\
          r"^Provides for consideration of \[BILL",r"^Provides for consideration of the Senate Amendment",\
          r"^Provides for consideration", r"^Provides for( \w+)? \[TIME\] of.*? debate",\
          r"^Pursuant to \[RESOLUTION\]", r"^Pursuant to a previous( \w+)? order", r"Pursuant to clause",\
          r"^Pursuant to the order of",\
          r"^proceeded with.*? debate",r"^Pursuant to the provisions of \[RESOLUTION\]",r"^QUESTION OF CONSIDERATION",\
          r"^QUESTION OF CONSIDERATION",r"^QUESTION OF THE PRIVILEGES OF THE HOUSE",r"^Received in the Senate",\
          r"^received in the senate[\,\.] read",\
          r"^Referred",r"^Resolution agreed to in Senate",r"Resolution provides for( \w+)? \[TIME\] of general debate",\
          r"(Rule|Resolution) provides for( \w+)? \[TIME\] of general debate","debate",\
          r"^Resolution provides for consideration of( \w+)? \[BILL",r"^Resolving differences",r"^Rule provides for consideration",\
          r"^Rules Committee Resolution",r"^Ruling of the Chair",r"^Second cloture", r"See( \w+)\[BILL\] ",
          r"^Senate Committee.*? discharged",r"^Senate agreed to.*? amendment",r"^Senate agreed to conference report",\
            r"Senate agreed to",r"Senate appointed conferee",r"Senate concurred in", r"Senate disagree. to.*? House amendment",
            r"Senate insists on its amendment",r"^Senate passed companion measure", r"Senate returned papers", 
            r"Senate vitiated previous",
            r"Star Print ordered",r"The Chair announced",r"^The Chair","postponed",r"The Committee of the Whole",
            "The Committee of the Whole proceeded",
            r"The Committee rose informally",r"The House proceeded with.*? \[TIME\] of debate",
            r"The House resolved into Committee of the Whole",
            r"The Speaker appointed (additional )?conferees",r"^The amendment.*? printed",r"^The amendment.*? recommended by",
            r"The committee.*? (amendment|substitute) agreed to",r"^The previous question.*? was ordered",
            r"The (resolution|rule) makes in order",
            r"The (resolution|rule) provides for.*? of debate",
            r"The (resolution|rule).*? provides for (the )?(further )?consideration of", 
            r"The (resolution|rule) provides that an amendment.*? shall be considered as adopted",
            r"The (resolution|rule) waives all points of order",r"The resolution waives clause 6",
            r"The (resolution|rule) provides that the bills shall be considered as read",
            r"The (resolution|rule) provides.*? motion.? to recommit per bill"
            r"The (resolution|rule) provides that an amendment.*? shall be considered as an original bill",
            r"^UNANIMOUS CONSENT",r"^UNFINISHED BUSINESS","60 votes", "five.minute rule",
            r"Upon reconsideration,.*? cloture",r"^VACATING DEMAND",r"^VACATING PROCEEDINGS",r"^Veto message",r"^WORDS TAKEN DOWN",
            r"Without objection, the Chair laid",r"points? of order",r"\[amendment agreed to in Senate\]",r"\[amendment.*? modified]",
            r"amendments? offered by",r"amendment.*?\b\w+\b(?<!\bnot) agreed to",r"amendment.*? not agreed to",r"\[AMENDMENT\].*? failed",
            r"\[AMENDMENT\] fell",
            r"\[AMENDMENT\].*? withdrawn",r"\[AMENDMENT\] modified",r"\[AMENDMENT\] motion.*? to reconsider the vote",
            r"motion to table amendment",r"motion to waive.*? budgetary discipline",
            r"\[AMENDMENT\] raised a point of order",
            r"\[AMENDMENT\] referred to",r"\[AMENDMENT\] second cloture",
            "Hearings held.",r"\[COMMITTEE\] discharged",r"\[COMMITTEE\].*? consideration held.",
            r"\[COMMITTEE\].*? reported by \[SENATOR\]", r"\[REPRESENTATIVE\] asked unanimous consent",r"\[BILLS\]",r"\[RESOLUTION\]"
            r"\[REPRESENTATIVE\] appealed the ruling of the (c|C)hair",r"^\[COMMITTEE\]",r"^\[REPRESENTATIVE\]",
            r"^\[AMENDMENT\]",r"\[RESOLUTIONS\]",r"\[LAW\]",r"\[BILL\]",r"\[RESOLUTIONS\]",r"\[RULE\]"
            r"\[REPRESENTATIVE\] moved that the House (agree|concur)",
            r"\[REPRESENTATIVE\] moved that the House (agree|concur|insist|disagree)",r"subcommittee","joint"
            r"\[REPRESENTATIVE\] moved to table the motion",
            r"\[REPRESENTATIVE\] raised a point of order",
            "^On motion","suspend the rules","read twice","instructions","by yea-nay vote",
            "withdrawn","Rule",r"print","conference report",
            "committee","unanimous consent","voice vote","cloture","closed rule","pursuant to",
            "unfinished","in the nature of a substitute","waive"}

# categorizes an action
def categorize(action):
  out=[]
  for regex in regexes:
    if re.search(regex,action.lower() if regex.islower() else action):
      out.append(regex)
  out.extend
  return out

# Makes categories out of these actions 
def make_categories(actions,categorize_f):
  categories=defaultdict(list)
  for action in actions:
    for category in categorize_f(action):
      categories[category].append(action)
  return dict(categories)
if __name__=="__main__":  
  parser = argparse.ArgumentParser(description="makes generics by manually stripping out names and combining actions with small edit distance")
  parser.add_argument("-d","--preprocessing_dir",type=str,default="./outputs/preprocess0", help="the directory for this preprocessing run")
  parser.add_argument("--threshold","-t",type=float,default=1/7,help="the max value of threshold*max(action1 length,action 2 length) for which the two actions will have the same generic")
  args,unknown = parser.parse_known_args()
  df = pd.read_csv(os.path.join(args.preprocessing_dir,"data_no_generics.csv"))
  actions = list(df["action"])
  with open(os.path.join(args.preprocessing_dir,"action_committee_map.json"),"r") as file:
    action_committee_map = json.load(file)
  categories_dict = make_categories(actions,lambda x:categorize(x)+action_committee_map[x])
  with open(os.path.join(args.preprocessing_dir,"categories_dict.json"),"w") as file:
    json.dump(categories_dict,file,indent=2)