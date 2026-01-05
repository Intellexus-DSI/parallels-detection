import os
from azure.storage.fileshare import ShareServiceClient
from azure.core.credentials import AzureSasCredential
from tqdm import tqdm

# ============================================================================
# CONFIGURATION
# ============================================================================
ACCOUNT_URL = "https://intlxresearchstorage.file.core.windows.net"
SAS_TOKEN = ""
SHARE_NAME = "intlx-gpu-fs"
AZURE_FILE_PATH = "data/05_clean_data/00_tibetan/Tibetan_1.jsonl"

# Local path where we want to save the file
LOCAL_OUTPUT_DIR = "data/05_clean_data/00_tibetan"
LOCAL_OUTPUT_FILE = os.path.join(LOCAL_OUTPUT_DIR, "Tibetan_1.jsonl")

def download_file_to_disk():
    # 1. Create local directory if it doesn't exist
    if not os.path.exists(LOCAL_OUTPUT_DIR):
        print(f"Creating local directory: {LOCAL_OUTPUT_DIR}")
        os.makedirs(LOCAL_OUTPUT_DIR)

    # 2. Connect to Azure
    print(f"Connecting to Azure Share: {SHARE_NAME}...")
    service_client = ShareServiceClient(account_url=ACCOUNT_URL, credential=AzureSasCredential(SAS_TOKEN))
    share_client = service_client.get_share_client(SHARE_NAME)
    file_client = share_client.get_file_client(AZURE_FILE_PATH)

    # 3. Get file size for progress bar
    props = file_client.get_file_properties()
    file_size = props.size
    print(f"Downloading '{AZURE_FILE_PATH}' ({file_size / 1024 / 1024:.2f} MB)...")

    # 4. Download and save to disk
    with open(LOCAL_OUTPUT_FILE, "wb") as file_handle:
        download_stream = file_client.download_file()
        # Use tqdm for a progress bar
        with tqdm(total=file_size, unit='B', unit_scale=True, desc="Saving to disk") as pbar:
            for chunk in download_stream.chunks():
                file_handle.write(chunk)
                pbar.update(len(chunk))

    print("\n" + "="*50)
    print(f"✓ File saved successfully to:\n  {os.path.abspath(LOCAL_OUTPUT_FILE)}")
    print("="*50)

if __name__ == "__main__":
    download_file_to_disk()