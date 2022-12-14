mport os
from pathlib import Path

from azure.storage.blob import BlobClient, ContainerClient

storage_account = os.environ["AZURE_STORAGE_ACCOUNT"]
account_url = f"https://{storage_account}.blob.core.windows.net"
container_name = "output"
credential = os.environ["AZURE_STORAGE_SAS_TOKEN"]
cc = ContainerClient(
    account_url,
    container_name=container_name,
    credential=credential,
)


for blob in cc.list_blobs():
    name = blob.name
    pname = Path(name)
    if name.startswith("wofs") and pname.stem.startswith('evi'):
        input_blob = BlobClient(account_url, container_name, blob.name, credential=credential)
        blob_url = input_blob.url
        new_name = str(pname.parent / Path('wofs_' + '_'.join(pname.stem.split('_')[1:]) + '.tif'))
        output_blob = BlobClient(account_url, container_name, new_name, credential=credential)
        #output_blob.upload_blob_from_url(blob_url, overwrite=True)
        input_blob.delete_blob()
        print(blob.name)

