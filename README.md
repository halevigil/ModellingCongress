# Modelling-Congress

Modelling the path of bills through congress, which actions on a bill are likely to come after which others:

https://github.com/user-attachments/assets/e47f7686-b91d-44e3-9d74-54c9900135bd

Explore for yourself:
https://modellingcongress-hidden-breeze-6910.fly.dev/

# Explanation

The public congress dataset stores a list of actions that take place in each congressional term and their associated bill. In this project I analyze this data, training a model to predict the future actions of a bill from its past actions.

To do this, I first scrub out all the specific information from each action - congressperson names, committee names, amounts of time, bill numbers, etc. - and combine similar actions to create a "generic" version of each action. I do this in two steps. First, in make_generics.py, I manually scrub out specific information via regex and combine actions which have low edit distance. Then, with the resultant generic names, in llm_refinement I ask a LLM to substitute specific information in these generic names with placeholders, so for example "Senator Schumer" becomes "\[SENATOR\]". With the resultant "refined" versions, I then again combine generics with low edit distance, to get a final list of generics and which generic each action corresponds to.

In addition to these generics, I create categories of actions manually through regexes in the categorize script.

Then I make the data ready for the predicition model. I train a simple one-layer neural network which has 6 vectors as input:

-   cumulative_generics , which is the sum of the one-hot vectors representing the generics a bill has previously seen
-   recent_generics , which is a weighted moving average of the one-hot vectors representing the generics a bill has previously seen
-   cumulative_categories , which is the sum of the one-hot vectors representing the generics a bill has previously seen
-   recent_categories , which is a weighted moving average of the many-hot vectors representing the categories a bill has previously seen
-   a one-hot vector representing the current term
-   a (potentially zero) one-hot vector representing the chamber

As output, it gives a probability vector representing the generic of the next action, and a probability vector representing the category of the next action. The former is enforced via softmax, and the second via sigmoid.

# Instructions

If you'd like to build the model youself:

1. Download the dataset from each congress term: https://legiscan.com/US/datasets. Unzip them all into the data folder:

```text
├── data
│   ├── 2009-2010_112th_Congress/
│   ├── 2011-2012_112th_Congress/
│   ├── 2013-2014_112th_Congress/
│   ├── 2015-2016_112th_Congress/
│   ├── 2017-2018_112th_Congress/
│   ├── 2019-2020_112th_Congress/
│   ├── 2021-2022_112th_Congress/
│   ├── 2023-2024_112th_Congress/
│   └── 2025-2026_112th_Congress/
```

2. Add your open api key to the environment variables as OPENAI_API_KEY. You can add it to a dotenv file in the project root and it will load it automatically

3. Run run.py with argument --preprocessing_dir to the preprocessing directory that you'd like. Likely, it will be something like outputs/preprocess
