# Refine generics using llm
import random
import numpy as np
import json
import random
import os
import shutil
import dotenv
import openai
import time
import argparse
import re

if __package__:
  from .make_generics import edit_distance_below
else:
  from make_generics import edit_distance_below

# creates input for llm
def llm_input(generics):
  generics_s="\n".join(generics)

  return f"""I am analyzing a dataset full of Congressional actions. I want to refine each action to clean them up and remove the names and descriptions of congresspeople, bills, amendments, times, votes, rules and committees.
Some notes to help you: 
H.R. and S. are bills. H.Res., S.Res. and H./S.Con.Res are resolutions. Amdt. are amendments.
Default to removing everything in parentheses (and the parentheses themselves) unless they are crucial for understanding the action. 
Remove the names of committees and subcommittees (replacing them with [COMMITTEE] and [SUBCOMMITTEE] respectively) except for the Committee on the Whole, the Rules Committee, the Committee on Standards of Official Conduct and any other of these types of committees that do not review bills but have some other functions.
Correct weird capitalization, spelling and punctuation.
The only times and vote numbers you'd keep are where they refer to specific rules, like a 60 vote threshold, five-minute rule, etc.  

Here are some example refinements you should base your refinement off of:
Action: Passed Senate with amendments by Unanimous Consent. (consideration: CR S1773-1774; text of committee substitute as amended: CR S1773)
Refinement: Passed Senate with amendments by Unanimous Consent.
Action: Rule provides for consideration of H.R. 273 with 1 hour of general debate. Previous question shall be considered as ordered without intervening motions except motion to recommit with or without instructions. Measure will be considered read. Bill is closed to amendments. Section 2 of the resolution provides that during any recess or adjournment of not more than three days, the Speaker or his designee may reconvene the House at a time other than that previously appointed, within the limits of clause 4, section 5, article I of the Constitution. Section 3 of the resolution authorizes the Speaker to entertain motions to suspend the rules through the legislative day of February 15, 2013, relating to a measure condemning the government of North Korea and its February 12, 2013 test of a nuclear device. Section 4 provides that on any legislative day from February 16, 2013 through February 22, 2013: (a) the Journal of the proceedings of the previous day shall be considered as approved; (b) the Chair may adjourn the House to meet at a date and time within the limits of clause 4, section 5, article I of the Constitution. Section 5 authorizes the Speaker to appoint Members to perform the duties of the Chair for the duration of the period addressed by section 4 as though under clause 8(a) of rule I.
Refinement: Rule provides for consideration of [BILL] with [TIME] of general debate. Previous question shall be considered as ordered without intervening motions except motion to recommit with or without instructions. Measure will be considered read. Bill is closed to amendments. [SECTION] of the resolution provides that during any recess or adjournment of not more than [TIME PERIOD], the Speaker or his designee may reconvene the House at a time other than that previously appointed, with the limits of [CLAUSE], [SECTION], [ARTICLE] of the Constitution. [SECTION] of the resolution authorizes the Speaker to entertain motions to suspend the rules through the legislative day of [DATE], relating to a measure [DESCRIPTION].
Action: Forwarded by Subcommittee to Full Committee by the Yeas and Nays: 18 - 0 and 1 Present.
Refinement: Forwarded by Subcommittee to Full Committee by the Yeas and Nays.
Action:ORDER OF PROCEDURE - Mr. Nunn (IA) asked unanimous consent that it be in order at any time through July 20, 2023, to consider the following joint resolutions in the House if called up by the Chair of the Committee on Foreign Affairs or his designee: H. J. Res. 68, H. J. Res. 70, H. J. Res. 71, or H. J. Res. 74; that each such joint resolution be considered as read; that the previous question be considered as ordered on each such joint resolution to final passage without intervening motion except for debate as follows: for H. J. Res. 68, 30 minutes equally divided among and controlled by Mr. McCaul, Mr. Meeks, and Mrs. Boebert or their respective designees; for H. J. Res. 70 and H. J. Res. 74, 30 minutes equally divided among and controlled by Mr. McCaul, Mr. Meeks, and Mr. Gosar or their respective designees; for H. J. Res. 71, 30 minutes equally divided among and controlled by Mr. McCaul, Mr. Meeks, and Mr. Crane or their respective designees; and that the provisions of section 202 of the                   National Emergencies Act (50 U .S.C. 1622) shall not apply                  to each such jo int resolution through July 20, 2023. Agreed to without objection. 
Refinement:ORDER OF PROCEDURE - [REPRESENTATIVE] asked unanimous consent that it be in order at any time through [DATE], to consider the following joint resolutions in the House if called up by the Chair of the Committee on Foreign Affairs or his designee: [RESOLUTIONS]; that each such joint resolution be considered as read; that the previous question be considered as ordered on each such joint resolution to final passage without intervening motion except for debate as follows: for [RESOLUTION], [TIME] equally divided among and controlled by Mr. McCaul, Mr. Meeks, and Mrs. Boebert or their respective designees; for H. J. Res. 70 and H. J. Res. 74, 30 minutes equally divided among and controlled by [REPRESENTATIVES] or their respective designees; for [RESOLUTION], [TIME] equally divided among and controlled by [REPRESENTATIVES] or their respective designees; and that the provisions of [SECTION] of the [LAW] shall not apply to each such joint resolution through [DATE]. Agreed to without objection. 
Action: GENERAL DEBATE - The Committee of the Whole proceeded with one hour of general debate on H.R. 1732. Pursuant to the provisions of H. Res. 231, the rule makes in order as original text for the purpose of amendment in the nature of a substitute consisting of the text Rules Committee Print 114-13 modified by the amendment printed in Part A of the Rules Committee report.
Refinement: GENERAL DEBATE - The Committee of the Whole proceeded with [TIME] of general debate on [BILL]. Pursuant to the provisions of [BILL], the rule makes in order as original text for the purpose of amendment in the nature of a substitute consisting of the text [PRINT] modified by the amendment printed in [PART] of the Rules Committee report.
Action: HR 1309 was forwarded to full committee by voice vote as amended by the Subcommittee on Insurance, Housing and Community Opportunity.
Refinement: [BILL] was forwarded to full committee by voice vote as amended by [SUBCOMMITTEE].
Action: In lieu of the amendment in the nature of a substitute recommended by the Committee on Financial Services now printed in the bill, it shall be in order to consider as an original bill for the purpose of amendment under the five-minute rule an amendment in the nature of a substitute consisting of the text of Rules Committee Print 114-42.
Refinement: In lieu of the amendment in the nature of a substitute recommended by [COMMITTEE] now printed in the bill, it shall be in order to consider as an original bill for the purpose of amendment under the five-minute rule an amendment in the nature of a substitute consisting of the text of Rules Committee [PRINT].
Action: In lieu of the amendment in the nature of a substitute recommended by the Committee on Financial Services now printed in the bill, it shall be in order to consider as an original bill for the purpose of amendment under the five-minute rule an amendment in the nature of a substitute consisting of the text of Rules Committee Print 114-42.
Refinement: In lieu of the amendment in the nature of a substitute recommended by [COMMITTEE] now printed in the bill, it shall be in order to consider as an original bill for the purpose of amendment under the five-minute rule an amendment in the nature of a substitute consisting of the text of Rules Committee [PRINT].
Action: Introduced in the Senate. Pursuant to Public Law 117-71, Ordered Placed on Senate Legislative Calendar under General Orders. Calendar No. 184.
Refinement: Intorduced in the Senate. Pursuant to [LAW], ordered placed on Senate Legislative Calendar under General Orders. Calendar [NUMBER].
Action: It shall be in order at any time through the legislative day of April 29, 2017, for the Speaker to entertain motions that the House suspend the rules.
Refinement: Action: It shall be in order at any time through the legislative day of [DATE], for the Speaker to entertain motions that the House suspend the rules.
Action: MOMENT OF SILENCE - The House observed a moment of silence in memory of 12 Marines who lost their lives in a January 14 training accident in Hawaii.
Refinement: MOMENT OF SILENCE - The House observed a moment of silence [DESCRIPTION].
Action: MOTION OFFERED - Pursuant to the provisions of H. Res. 742, the Chair recognized Mr. Shimkus for a motion.
Refinement: MOTION OFFERED - Pursuant to the provisions of H. Res. 742, the Chair recognized [REPRESENTATIVE] for a motion.
Action: Motion by Senator McConnell to agree to the House amendment to the Senate bill made in Senate. (consideration: CR S4577)
Refinement: Motion by [SENATOR] to agree to the House amendment to the Senate bill made in Senate.
Action: Motion by Senator Reid to concur in the House amendment to the bill (S. 2038) with an amendment (SA 1940) withdrawn in Senate by Unanimous Consent. (consideration: CR S1981)
Refinement: Motion by [SENATOR] to concur in the House amendment to the bill [BILL] with an amendment [AMENDMENT] withdrawn in Senate by Unanimous Consent.
Action: Mr. Andrews moved to recommit with instructions to Energy and Commerce, Ways and Means, and Education and the Workforce. (consideration: CR H321-322; text: CR H321)
Refinement: [REPRESENTATIVE] moved to recommit with instructions to [COMMITTEES].
Action: Mr. Smith (WA) asked unanimous consent to take from the Speaker's table, the bill S. 1790, and ask for its immediate consideration in the House; to strike out all after the enacting clause of such bill and insert in lieu thereof the provisions of H.R. 2500 as passed by the House; to pass the Senate bill, as amended; and to insist on the House amendment thereto and request a conference with there Senate thereon. Agreed to without objection.
Refinement: [REPRESENTATIVE] asked unanimous consent to take from the Speaker's table, [BILL], and ask for its immediate consideration in the House; to strike out all after the enacting clause of such bill and insert in lieu thereof the provisions of [BILL] as passed by the House; to pass the Senate bill, as amended; and to insist on the House amendment thereto and request a conference with there Senate thereon. Agreed to without objection.
Action: POSTPONED PROCEEDINGS - The Chair put the question on adoption of the Waters amendment and by voice vote, announced that the ayes had prevailed. Mrs. Biggert demanded a recorded vote and the Chair postponed further proceedings on the question of adoption of the amendment until later in the legislative day.
Refinement: POSTPONED PROCEEDINGS - The Chair put the question on adoption of the [REPRESENTATIVE] amendment and by voice vote, announced that the ayes had prevailed. [REPRESENTATIVE] demanded a recorded vote and the Chair postponed further proceedings on the question of adoption of the amendment until later in the legislative day.
Action: Pursuant to the rule the committee rose.
Refinement: Pursuant to the rule the committee rose.
Action: Referred to House Science and Technology
Refinement: Referred to [COMMITTEE].
Action: Referred to the Subcommittee on Regulations and Healthcare .
Refinement: Referred to [SUBCOMMITTEE].
Action: Rules Committee Resolution H. Res. 787 Reported to House. The resolution provides for one hour of debate on each bill. The resolution makes in order only the further amendment to H.R. 4566 printed in the report. No further amendments on either bill are made in order. Section 3 of the resolution makes it in order to consider any resolution reported from the Rules Committee on the day it is reported through the legislative day of March 23, 2018. Section 4 grants suspension authority on the legislative days of March 22, 2018 and March 23, 2018. Section 5 of the resolution amends section 3(a) of H. Res. 5.
Refinement: Rules Committee Resolution [RESOLUTION] reported to House. The resolution provides for [TIME] of debate on each bill. The resolution makes in order only the further amendment to [BILL] printed in the report. No further amendments on either bill are made in order. [SECTION] of the resolution makes it in order to consider any resolution reported from the Rules Committee on the day it is reported through the legislative day of [DATE]. [SECTION] grants suspension authority on the legislative days of [DATES]. [SECTION] of the resolution amends [SECTION] of [RESOLUTION].
Action: S.Amdt.845 Referred to the Committee on Rules and Administration.
Refinement: [AMENDMENT] referred to the Committee on Rules and Administration.
Action: See also S.Con.Res. 37, S.Con.Res. 40, H.Con.Res. 112.
Refinement: See also [RESOLUTIONS].
Action: Referred to the Subcommitte on Trade.
Refinement: Referred to [SUBCOMMITTEE].
Action: Resolving differences -- Senate actions: Senate agreed, under the order of 12/12/2024, having achieved 60 votes in the affirmative, to the House amendment to S. 4367 by Yea-Nay Vote. 97 - 1. Record Vote Number: 327.
Refinement: Resolving differences -- Senate actions: Senate agreed, under the order of [DATE], having achieved 60 votes in the affirmative, to the House amendment to S. 4367 by Yea-Nay Vote.
Action: Rules Committee Resolution H. Res. 985 Reported to House. Rule provides for consideration of H.R. 50 and H.R. 3281. Rule provides for consideration of H.R. 50 under a structured rule and H.R.3281 under a closed rule. Each measure is allowed one motion to recommit with or without instructions.
Refinement: Rules Committee Resolution [RESOLUTION] reported to House. Rule provides for consideration of [BILLS]. Rule provides for consideration of [BILL] under a structured rule and [BILL] under a closed rule. Each measure is allowed one motion to recommit with or without instructions.
Action: All points of order against consideration are waived except those arising under clause 9 or 10 of rule XXI. It shall be in order to consider as an original bill for the purpose of amendment under the five-minute rule the amendment in the nature of a substitute recommended by the Committee
Refinement: All points of order against consideration are waived except those arising under clause 9 or 10 of rule XXI. It shall be in order to consider as an original bill for the purpose of amendment under the five-minute rule the amendment in the nature of a substitute recommended by the Committee.
Action: An errata sheet on written report No. 112-264 was printed to include Minority views.
Refinement: An errata sheet on written report [NUMBER] was printed to include Minority views.
Action: By Senator Blunt from Committee on Rules and Administration filed written report. Report No. 114-112.
Refinement: By [SENATOR] from Committee on Rules and Administration filed written report. Report [NUMBER].
Action: Cloture motion on the motion to waive the points of order under section 303 of the Congressional Budget Act of 1974, any amendments thereto and motions thereon presented in Senate.  
Refinement: Cloture motion on the motion to waive the points of order under [SECTION] of [LAW], any amendments thereto and motions thereon presented in Senate.
Action: Committee on Foreign Relations. Reported by Senator Kerry with amendments and with a preamble. With written report No. 112-27. Minority views filed."
Refinement: [COMMITTEE]. Reported by [SENATOR] with amendments and with a preamble. With written report [NUMBER]. Minority views filed.
Action: Committee on Agriculture, Nutrition, and Forestry. Hearings held. Hearings printed: S.Hrg. 119-55.
Refinement: [COMMITTEE]. Hearings held. Hearings printed: [HEARING].
Action: Rule provides for consideration of H.R. 1018 with 1 hour of general debate. Previous question shall be considered as ordered without intervening motions except motion to recommit with or without instructions. Measure will be considered read. Specified amendments are in order. The resolution waives all points of order against consideration of the bill except for clauses 9 and 10 of rule XXI. The amendment in the nature of a substitute recommended by the Committee on Natural Resources shall be considered as adopted. The resolution waives all points of order against provisions of the bill, as amended. This waiver does not affect the point of order available under clause 9 of rule XXI.
Refinement: Rule provides for consideration of [BILL] with 1 hour of general debate. Previous question shall be considered as ordered without intervening motions except motion to recommit with or without instructions. Measure will be considered read. Specified amendments are in order. The resolution waives all points of order against consideration of the bill except for clauses 9 and 10 of rule XXI. The amendment in the nature of a substitute recommended by [COMMITTEE] shall be considered as adopted. The resolution waives all points of order against provisions of the bill, as amended. This waiver does not affect the point of order available under clause 9 of rule XXI.
Action: Considered as unfinished business. H.R. 6513 - "An Act to amend the Help America Vote Act of 2002 to confirm the requirement that States allow access to designated congressional election observers to observe the election administration procedures in congressional elections." (consideration: CR H5800)
Refinement: Considered as unfinished business. [BILL] - [DESCIRPTION].
Action: ORDER OF PROCEDURE - Mr. George Miller (CA) asked unanimous consent that, during proceedings today in the House and in the Committee of the Whole, the Chair be authorized to reduce to two minutes the minimum time for electronic voting on any question that otherwise could be subjected to five-minute voting under clause 8 or 9 or rule 20 or under clause 6 of rule 18.
Refinement: ORDER OF PROCEDURE - [REPRESENTATIVE] asked unanimous consent that, during proceedings today in the House and in the Committee of the Whole, the Chair be authorized to reduce to [TIME] the minimum time for electronic voting on any question that otherwise could be subjected to five-minute voting under clause 8 or 9 or rule 20 or under clause 6 of rule 18.
Action: Considered under the provisions of H.Res. 391 and section 1002(e) of the Continuing Appropriations Act, 2014. (consideration: CR H6872-6876)
Refinement: Considered under the provisions of [BILL] and [SECTION] of [LAW].
Action: DEBATE - The House proceeded with one hour of debate on the conference report H. Rept. 112-331 for consideration under the provisions of H. Res. 500.
Refinement: DEBATE - The House proceeded with [TIME] of debate on the conference report [REPORT] for consideration under the provisions of [RESOLUTION].
Action: For H.R. 6, one hour of general debate and amendments are confined to those printed in part A of the report accompanying this resolution. For H.R. 3301, one hour of general debate and amendments are confined to those printed in part B of the report accompanying this resolution.
Refinement: For [BILL], [TIME] of general debate and amendments are confined to those printed in [PART] of the report accompanying this resolution. For [BILL], [TIME] of general debate and amendments are confined to those printed in [PART] of the report accompanying this resolution.
Action: The Speaker appointed additional conferees - from the Committee on Ways and Means, for consideration of secs. 31101, 31201, and 31203 of the House amendment, and secs. 51101, 51201, 51203, 52101, 52103-05, 52108, 62001, and 74001 of the Senate amendment, and modifications committed to conference: Brady (TX), Reichert, and Levin. (consideration: CR H8278)
Refinement: The Speaker appointed additional conferees - from [COMMITTEE], for consideration of [SECTIONS] of the House amendment, and [SECTIONS] of the Senate amendment, and modifications committed to conference: [REPRESENTATIVES].
Action: The Speaker appointed conferees - from the Committee on House Administration for consideration of Subtitle H of Title V of the Senate amendment, and modifications committed to conference.
Refinement: The Speaker appointed conferees - from the Committee on House Administration for consideration of [SUBTITLE] of the Senate amendment, and modifications committed to conference.
Action: The previous question on each measure is considered ordered without intervening motions except one hour of debate and a motion to recommit. H. Res. 316 and H. Con. Res. 30 are adopted.
Refinement: The previous question on each measure is considered ordered without intervening motions except [TIME] of debate and a motion to recommit. [BILLS] are adopted.
Action: DEBAT - Pursuant to the provisions of H.Res. 281, the Committee of the Whole proceeded with 10 minutes of debate on the Rahall of West Virginia amendment.
Refinement: DEBATE - Pursuant to the provisions of [BILL], the Committee of the Whole proceeded with [TIME] of debate on the [REPRESENTATIVE] amendment.
Action: Senate agreed to conference report to accompany S. 524 by Yea-Nay Vote. 92 - 2. Record Vote Number: 129.
Refinement: Senate agreed to conference report to accompany [BILL] by Yea-Nay Vote. 
Action: The resolution provides for consideration of H.R. 788 under a structured rule, with one hour of general debate. The resolution provides for consideration of H.J. Res. 98 and S.J. Res. 38 under a closed rule, with one hour of general debate on each joint resolution. Also, the resolution provides for a motion to recommit on H.R. 788 and H.J. Res. 98. A motion to commit on S.J. Res. 38.
Refinement: The resolution provides for consideration of [BILL] under a structured rule, with [TIME] of general debate. The resolution provides for consideration of [RESOLUTIONS] under a closed rule, with [TIME] of general debate on each joint resolution. Also, the resolution provides for a motion to recommit on [BILL] and [RESOLUTION]. A motion to commit on [RESOLUTION].
Action: On retaining Title II (Department of Veterans Affairs) Agreed to by recorded vote: 409 - 1 (Roll no. 416).
Refinement: On retaining [LAW] agreed to by recorded vote.
Action: Introduced in the Senate. Read twice. Ordered Placed on Senate Legislative Calendar under General Orders. Calendar No. 124.
Refinement: Introduced in the Senate. Read twice. Ordered placed on Senate Legislative Calendar under General Orders. Calendar [NUMBER].
Action: On motion to agree in the Senate amendments numbered 2 and 3, and agree in Senate amendment numbered 1 with an amendment agreed to by the Yeas and Nays.  
Refinement: On motion to agree in the Senate amendments [NUMBERS], and agree in Senate amendment [NUMBER] with an amendment agreed to by the Yeas and Nays.
Action: ORDER OF PROCEDURE - Mr. Diaz-Balart asked unanimous consent for an additional 10 minutes of debate on each side of the aisle. Agreed to without objection.
Action: ORDER OF PROCEDURE - [REPRESENTATIVE] asked unanimous consent for an additional [TIME] of debate on each side of the aisle. Agreed to without objection.

Now, please come up with a refinement for the following actions. Please only give me the refinements, one per line in the same order as given in, nothing else in your response. If there is just one action, just give me the refinement for that action:
{generics_s}"""



# Creates the batches fr llm refinement
# Returns paths to the created batches
def create_batches(actions,input_dir,batch_size):
  if not batch_size%5==0:
    raise Exception("batch size must be divisible by 5")
  generics_batches = np.array_split(actions, range(batch_size, len(actions), batch_size))
  action_i=0
  paths=[]
  for batch_i,batch in enumerate(generics_batches):
    minibatches=np.array_split(batch,range(5,len(batch),5))
    batch_path=os.path.join(input_dir,f"batch{batch_i}.jsonl")
    with open(batch_path,"w") as file:
      for minibatch in minibatches:
        input=llm_input(minibatch)
        json.dump({"custom_id":f"generics {action_i}-{action_i+len(minibatch)}","url":"/v1/responses","method":"POST","body":{"input":input,"model":"gpt-5-mini"}}, file)
        file.write("\n")
        action_i+=len(minibatch)
    paths.append(batch_path)
  return len(paths)

def run_batches(input_dir,output_dir,batch_is=None):
  dotenv.load_dotenv()
  client = openai.Client(api_key=os.environ["OPENAI_API_KEY"])

  n_batches = len(os.listdir(input_dir))
  if not batch_is:
    batch_is = range(n_batches)
  input_paths = [os.path.join(input_dir,f"batch{batch_i}.jsonl") for batch_i in batch_is]
  j=0
  while j<len(batch_is):
    batch_i = batch_is[j]
    path = input_paths[j]
    statuses=[batch.status for batch in client.batches.list()]
    while "cancelling" in statuses or "in_progress" in statuses or "finalizing" in statuses:
      print("waiting before starting batch... ")
      time.sleep(30)
      statuses=[batch.status for batch in client.batches.list()]
    with open(path,"rb") as file:
      batch_file=client.files.create(file=file,purpose="batch")
    batch=client.batches.create(input_file_id=batch_file.id,endpoint="/v1/responses",completion_window="24h",metadata={"batch_i":f"{batch_i}"})
    print(batch.id)
    failed=False
    while (status:=client.batches.retrieve(batch.id).status)!="completed" and status!="failed":
      time.sleep(20)
    if status=="failed":
      print("failed")
      time.sleep(30)
      continue
    else:
      output_file = os.path.join(output_dir,f"batch{batch_i}.jsonl")
      with open(output_file,"w") as file:
        file.write(client.files.content(file_id=client.batches.retrieve(batch_id=batch.id).output_file_id).text)
      print(f"finished batch {batch_i}")
    j+=1
    time.sleep(180)

def manual_refinement(name):
  if not re.search("five.minute.rule",name):
    name = re.sub(r"\\w+ (minutes)|(minute)|(hour)|(hours)","[TIME]",name)
  name = re.sub(r"\[TIME\]s","[TIME]",name)
  name = re.sub(r"H\. ?Res\. ?[0-9]+","[RESOLUTION]",name)
  name = re.sub(r"H\. ?R\. ?[0-9]+","[BILL]",name)
  name = re.sub(r"S\. ?[0-9]+","[BILL]",name)
  name = re.sub(r"S\ ?.Res\. ?[0-9]+","[RESOLUTION]",name)
  name = re.sub(r"[0-9]+ \[TIME\]","[TIME]",name)
  name = re.sub(r"(one|two|three|four|five|ten|thirty|sixty) \[TIME\]","[TIME]",name)
  name = re.sub(r"\[TIME\]s","[TIME]",name)
  name = re.sub(r"(No|no)\. [0-9]+", "[NUMBER]", name)
  name = re.sub(r"\(.*\)", "", name)
  name = re.sub(r"  +", " ", name)
  name=name.strip()
  return name

# Create refinements from actions
def create_refinements(actions):
  dotenv.load_dotenv()
  client = openai.Client(api_key=os.environ["OPENAI_API_KEY"])
  out=[]
  for action_minibatch in np.array_split(actions,range(5,len(actions),5)):
    out.extend(create_refinements_minibatch(action_minibatch))
  return out
  
# Create refinements from a minibatch (usually of size 5) of actions
def create_refinements_minibatch(actions):
  dotenv.load_dotenv()
  client = openai.Client(api_key=os.environ["OPENAI_API_KEY"])
  inpt = llm_input(actions)
  # Tries 5 times to come up with refinements, as someitmes the LLM will miss lines
  for i in range(5):
    response = client.responses.create(input=inpt,model="gpt-5")
    try:
      refinements= response.output[1].content[0].text.split("\n")
      assert(len(refinements)==len(actions))
      return refinements
    except:
      continue
  return actions
  

# Creates a refinement map from llm_output
# Returns the refinement map and all the minibatches that were incomplete
def create_refinement_map(actions,llm_output_dir):
  refinements=[None for i in range(len(actions))]
  incomplete=[]
  response_start=0
  refinement_map={}
  for path in [os.path.join(llm_output_dir,f"batch{batch_i}.jsonl") for batch_i in range(len(os.listdir(llm_output_dir)))]:
    with open(path,"r") as file:
      for response in file:
        text = json.loads(response)["response"]["body"]["output"][1]["content"][0]["text"]
        actions_minibatch = actions[response_start:response_start+5]
        refinements=[x for x in text.split("\n") if x!=""]
        # If not exactly 5 (minibatch size) refinements were returned, add to incomplete so it can be manually processed afterwards
        if len(refinements)!=5:
          response_start+=5
          incomplete.append((actions_minibatch,refinements))
          continue
        for action,refinement in zip(actions_minibatch,refinements):
          print(action,":",refinement)
          refinement_map[action]=refinement
        response_start+=5
  return refinement_map,incomplete
if __name__=="__main__":
  parser = argparse.ArgumentParser(description="uses llm to refine manual generics")
  parser.add_argument("-d","--preprocessing_dir",type=str,default="outputs/preprocess0", help="the directory for this preprocessing run")
  parser.add_argument("--batch_size",type=int,default=750)
  parser.add_argument("--overwrite",action="store_true")
  parser.add_argument("--batch_is",nargs="*")
  


  args,unknown = parser.parse_known_args()
  dotenv.load_dotenv()
  input_dir=os.path.join(args.preprocessing_dir,"llm_refinement_input")
  output_dir=os.path.join(args.preprocessing_dir,"llm_refinement_output")
  
  if args.overwrite:
    if os.path.exists(input_dir):
      shutil.rmtree(input_dir)
    if os.path.exists(output_dir):
      shutil.rmtree(output_dir)
  if not os.path.exists(input_dir):
    os.mkdir(input_dir)
  if not os.path.exists(output_dir):
    os.mkdir(output_dir)
    
  with open(os.path.join(args.preprocessing_dir,"generics_dict_manual.json"),"r") as file:
    generics_dict = json.load(file)
  generics=list(generics_dict.keys())
  create_batches(generics,os.path.join(args.preprocessing_dir,"llm_refinement_input"),args.batch_size)
  run_batches(os.path.join(args.preprocessing_dir,"llm_refinement_input"),os.path.join(args.preprocessing_dir,"llm_refinement_output"))
  refinement_map,incomplete = create_refinement_map(generics,os.path.join(args.preprocessing_dir,"llm_refinement_output"))
  for actions5, _ in incomplete:
    refinement_map.update(create_refinements(actions5))

  with open(os.path.join(args.preprocessing_dir,"refinement_map.json"),"w") as file:
    json.dump(refinement_map,file,indent=2)
