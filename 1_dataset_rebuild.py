import os
import pandas
import subprocess


"""1st step: Download and validate PDFs

Dataset provided by the Datapolitics -> dataset.csv

Defaults: The server is not stable, each time the download is launched, there are some "access denied" errors.

To solve this problem, our team propose a solution with the following steps:
1. curl the dataset.csv text version column to get the txt file
2. for each txt file, read the content, if the content is less than 500 characters, we abandon it
3. update the dataset.csv with the selected txt files, and create a new column call local_file to store the txt file at local computer.

:param dataset: the dataset.csv file
:param output_dir: the output directory to store the txt files
:param min_length: the minimum length of the text content, default 500
:return: the updated dataset.csv file

"""


def download_and_validate_pdfs(dataset, output_dir="./txt", min_length=500):
    """Download PDFs and validate their content"""
    os.makedirs(output_dir, exist_ok=True)

    valid_files = {}
    failed_downloads = []
    file_paths = {}  # New dict to store file paths

    for counter, (index, url) in enumerate(tqdm(dataset['text version'].items())):
        file_name = f"dataset_pdf_{counter}.txt"
        file_path = os.path.join(output_dir, file_name)

        try:
            subprocess.run(["curl", "-L", "-o", file_path, url],
                           check=True, capture_output=True)

            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                if len(content) >= min_length:
                    valid_files[counter] = index
                    file_paths[index] = file_name  # Store filename for valid files
                else:
                    os.remove(file_path)
                    failed_downloads.append(index)

        except (subprocess.CalledProcessError, IOError):
            failed_downloads.append(index)
            if os.path.exists(file_path):
                os.remove(file_path)

    # Update dataset
    dataset_valid = dataset[~dataset.index.isin(failed_downloads)].copy()
    dataset_valid['local_filename'] = dataset_valid.index.map(file_paths)

    assert len(os.listdir(output_dir)) == len(dataset_valid)

    return dataset_valid


if __name__ == "__main__":
    dataset = pandas.read_csv("dataset.csv")
    dataset_valid = download_and_validate_pdfs(dataset)
    dataset_valid.to_csv("dataset_valid.csv", index=False)
    print(f"Valid entries: {len(dataset_valid)}")
    print(dataset_valid[['text version', 'local_filename']].head())
    print("Done. now please proceed next step : 2_ner.py")
