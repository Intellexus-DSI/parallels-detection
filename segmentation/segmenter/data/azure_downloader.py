"""Azure file download functionality."""

import os
from pathlib import Path

from azure.storage.fileshare import ShareServiceClient
from azure.core.credentials import AzureSasCredential
from tqdm import tqdm


class AzureDownloader:
    """Downloads files from Azure File Share."""

    def __init__(self, account_url: str, sas_token: str, share_name: str):
        """Initialize Azure downloader.

        Args:
            account_url: Azure storage account URL
            sas_token: SAS token for authentication
            share_name: Name of the file share
        """
        self.account_url = account_url
        self.sas_token = sas_token
        self.share_name = share_name

    def download_file(self, azure_path: str, local_path: Path) -> None:
        """Download a file from Azure to local disk.

        Args:
            azure_path: Path to file in Azure file share
            local_path: Local path to save the file
        """
        # Create local directory if it doesn't exist
        local_path.parent.mkdir(parents=True, exist_ok=True)

        # Connect to Azure
        print(f"Connecting to Azure Share: {self.share_name}...")
        service_client = ShareServiceClient(
            account_url=self.account_url,
            credential=AzureSasCredential(self.sas_token),
        )
        share_client = service_client.get_share_client(self.share_name)
        file_client = share_client.get_file_client(azure_path)

        # Get file size for progress bar
        props = file_client.get_file_properties()
        file_size = props.size
        print(f"Downloading '{azure_path}' ({file_size / 1024 / 1024:.2f} MB)...")

        # Download and save to disk
        with open(local_path, "wb") as file_handle:
            download_stream = file_client.download_file()
            with tqdm(
                total=file_size, unit="B", unit_scale=True, desc="Saving to disk"
            ) as pbar:
                for chunk in download_stream.chunks():
                    file_handle.write(chunk)
                    pbar.update(len(chunk))

        print("\n" + "=" * 50)
        print(f"✓ File saved successfully to:\n  {local_path.absolute()}")
        print("=" * 50)
