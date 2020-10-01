import zipfile
from google_drive_downloader import GoogleDriveDownloader as gdd
from tqdm import tqdm
import requests

gdd.download_file_from_google_drive(file_id='1cQEjNIb11eOL4ZPKKvXPvdx9OVL324Zp',
                                    dest_path='./checkpoints.zip')
with zipfile.ZipFile("checkpoints.zip", 'r') as zip_ref:
    zip_ref.extractall(".")


