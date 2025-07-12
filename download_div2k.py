import os
import requests
from zipfile import ZipFile

DIV2K_URL = "https://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_HR.zip"
ZIP_FILE = "DIV2K_train_HR.zip"
EXTRACT_DIR = "training_data"

def download_dataset(url: str, zip_path: str):
    """Download the dataset zip if it doesn't exist locally."""
    if os.path.exists(zip_path):
        print("DIV2K dataset already downloaded.")
        return

    print("Downloading DIV2K dataset...")
    with requests.get(url, stream=True) as response:
        response.raise_for_status()
        with open(zip_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
    print("Download completed.")

def extract_zip(zip_path: str, extract_to: str):
    """Extract the zip file contents into a directory."""
    print("Extracting dataset...")
    with ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    print(f"Extraction complete. Files saved in: '{extract_to}'")

if __name__ == "__main__":
    download_dataset(DIV2K_URL, ZIP_FILE)
    extract_zip(ZIP_FILE, EXTRACT_DIR)
