# Finetune LLM Depression Data

This repository contains code for generating synthetic data for depression detection using Parameter-Efficient Fine-Tuning (PEFT) on Large Language Models.

## Data Structure

For the preprocessing scripts to work correctly, your data must be organized in the following folder structure:

```
daic_data/
├── labels/
│   ├── dev.csv
│   ├── test.csv
│   └── train.csv
│
└── transcripts/
    ├── 300_TRANSCRIPT.csv
    ├── 301_TRANSCRIPT.csv
    ├── ...
    ├── 491_TRANSCRIPT.csv
    └── 492_TRANSCRIPT.csv
```
