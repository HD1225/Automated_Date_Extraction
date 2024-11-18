#!/bin/bash

# if you want to process all the dataset
#1. download all the text file from datapolitics
python 1_dataset_rebuild.py
#2. do ner
python 2_ner.py --csv ./dataset_valid.csv
#3. llm reference
python 4_llm_reference.py
#4. clean_result
python 5_clean_date.py ./final_results_predicted.csv -o ./pipeline_result.csv

echo "Ready to calculate accuracy"

python 6_evaluation.py -i ./pipeline_result.csv -o ./pipeline_result.csv

echo "pipeline end,please check the result in pipeline_result.csv"