import zipfile
from google_drive_downloader import GoogleDriveDownloader as gdd
from tqdm import tqdm
import requests

url = "https://s3.amazonaws.com/ai2-vision-textbook-dataset/dataset_releases/tqa/tqa_train_val_test.zip"
response = requests.get(url, stream=True)

with open("tqa_train_Val_test.zip", "wb") as handle:
    for data in tqdm(response.iter_content()):
        handle.write(data)
with zipfile.ZipFile("tqa_train_Val_test.zip", 'r') as zip_ref:
    zip_ref.extractall(".")

gdd.download_file_from_google_drive(file_id='11QE4nwU3pVB_0Q5E45P-3wuuhcG1g3yH',
                                    dest_path='./jsons.zip')
with zipfile.ZipFile("jsons.zip", 'r') as zip_ref:
    zip_ref.extractall(".")


