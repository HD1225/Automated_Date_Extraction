import os
from transformers import AutoTokenizer,Pipeline,AutoModelForTokenClassification
import pandas
import torch
import argparse

"""2 step 
This script is used to extract the date information from the text files using the NER model.
result will be added to the dataset.csv file as a new column, this would facilitate the prompt at next step 

"""
# IF initializing NER model with problem
## Try : uninstall sentencepiece and reinstall it

def clean_data(data):
    cleaned = []
    for line in data:
        line = line.strip()  # 去除空格
        if not line or line.isspace() or set(line) <= {'-', '_', ' ', '—'} or '\x0c' in line:
            continue
        line = " ".join((line.split()))
        cleaned.append(line)

    return cleaned
def get_ner_date(file):
    # data cleaning
    with open(file, 'r', encoding='utf-8') as f:
        data = f.read().split('\n')
    data = clean_data(data)
    # data processing
    date_results = []
    results_total = nlp(data)
    for sublist in results_total:
        for item in sublist:
            if item['entity'] == 'DATE':
                date_results.append(item['word'])
    data_results = [i for i in date_results if 20>len(i)>5 and "à" not in i]
    data_results = data_results[:15]
    return data_results

if __name__ == "__main__":
    results = {}
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=str, help="The path to the txt file.")
    parser.add_argument("--model", type=str, help="The model name.",default="Jean-Baptiste/camembert-ner-with-dates")
    parser.add_argument("--csv", type=str, help="The path to the csv file.")
    parser.add_argument("--device", type=str, help="The device to use.",default="cpu")

    args = parser.parse_args()
    device = torch.device(args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForTokenClassification.from_pretrained(args.model).to(device)
    nlp = Pipeline('ner', model=model, tokenizer=tokenizer,aggregation_strategy='simple',device=device)


    file_list = os.listdir(args.file)
    print("it may take a while")
    for file in file_list:
        file_path = os.path.join(args.file, file)
        date_results = get_ner_date(file_path)
        results[file] = date_results
    # put the results into a dataframe
    df = pandas.read_csv(args.csv)
    df['date'] = df['local_filename'].map(results)
    csv_name = args.csv.split(".")[0] + "_ner.csv"
    df.to_csv(csv_name, index=False)



