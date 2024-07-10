import requests
import os
from tqdm import tqdm
import zipfile
import tarfile

RED = '\033[31m'
GREEN = '\033[32m'
YELLOW = '\033[33m'
BLUE = '\033[34m'
MAGENTA = '\033[35m'
CYAN = '\033[36m'
WHITE = '\033[37m'
RESET = '\033[0m'

LOG     = BLUE   + "[LOG]      " + RESET
ERROR   = RED    + "[ERROR]    " + RESET
WARNING = YELLOW + "[WARNING]  " + RESET
SUCCESS = GREEN  + "[SUCCESS]  " + RESET

def format_duration(seconds):
    days = seconds // (24 * 3600)
    seconds %= (24 * 3600)
    hours = seconds // 3600
    seconds %= 3600
    minutes = seconds // 60
    seconds %= 60
    
    duration_str = ""
    if days > 0:
        duration_str += f"{days} Day(s) "
    if hours > 0:
        duration_str += f"{hours} Hour(s) "
    if minutes > 0:
        duration_str += f"{minutes} Minute(s) "
    if seconds > 0:
        duration_str += f"{seconds} Second(s)"
    
    return duration_str.strip()

def remove_zip_tar_files(folder_path):
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if filename.endswith('.zip') or filename.endswith('.tar.gz'):
            print(f'Removing {filename}')
            os.remove(file_path)

def download_file(url, dest_folder):
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    filename = os.path.join(dest_folder, url.split('/')[-1])

    with open(filename, 'wb') as file, tqdm(
        desc=f'{LOG}Downloading {filename}',
        total=total_size,
        unit='B',
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for chunk in response.iter_content(chunk_size=1024):
            size = file.write(chunk)
            bar.update(size)
    
    return filename

def extract_file(filepath, dest_folder):
    if filepath.endswith('.zip'):
        with zipfile.ZipFile(filepath, 'r') as zip_ref:
            zip_info_list = zip_ref.infolist()
            total_files = len(zip_info_list)

            with tqdm(total=total_files, desc=f'{LOG}Extracting {filepath}', unit='file') as bar:
                for file_info in zip_info_list:
                    zip_ref.extract(file_info, dest_folder)
                    bar.update(1)
    elif filepath.endswith(('.tar.gz', '.tgz', '.tar')):
        with tarfile.open(filepath, 'r:*') as tar_ref:
            tar_info_list = tar_ref.getmembers()
            total_files = len(tar_info_list)

            with tqdm(total=total_files, desc=f'{LOG}Extracting {filepath}', unit='file') as bar:
                for file_info in tar_info_list:
                    tar_ref.extract(file_info, dest_folder)
                    bar.update(1)
    remove_zip_tar_files(dest_folder)
