# Modelling-Congress

Modelling the path of bills through congress, which actions on a bill are likely to come after which others.

See

https://github.com/user-attachments/assets/fbd8378c-6376-4ff6-88ef-73a31a45f2da

Explore for yourself:
https://modellingcongress-hidden-breeze-6910.fly.dev/

# Instructions

If you'd like to build the model youself:

1. Download the dataset from each congress term, and unzip them all into the data folder:

```text
.
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

2. Run run.py with argument --preprocessing_dir to the preprocessing directory that you'd like. Likely, it will be something like outputs/preprocess
