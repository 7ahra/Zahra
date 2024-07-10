import os
import json
import argparse
from utils.helpers import extract_file, download_file, LOG, WARNING, SUCCESS

def main(dataset):
    with open('utils/urls.json', 'r') as f:
        datasets = json.load(f)
    
    datasets_to_download = datasets if dataset == 'all' else {dataset: datasets[dataset]}

    if dataset not in datasets and dataset != 'all':
        print(f'{WARNING}Dataset "{dataset}" not found in datasets.json')
        return
    
    urls = datasets[dataset]
    dest_folder = './datasets'
    os.makedirs(dest_folder, exist_ok=True)

    for dataset_name, urls in datasets_to_download.items():
        for name, url in urls.items():
            print(f'{LOG}Starting download for {dataset_name} - {name}')
            dest = os.path.join(dest_folder, dataset_name)
            os.makedirs(dest, exist_ok=True)
            filepath = download_file(url, dest_folder=dest)
            print(f'{LOG}Extracting {filepath}')
            extract_file(filepath, dest)
            print(f'{SUCCESS}Completed extraction for {filepath}')
    
    print(f"{SUCCESS}Dataset downloaded and ready to train!")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Download and extract datasets.')
    parser.add_argument('dataset', choices=['ms_coco', 'imageNet', 'all'], default="all", help='The name of the dataset to download or "all" to download all datasets')
    args = parser.parse_args()
    
    main(args.dataset)