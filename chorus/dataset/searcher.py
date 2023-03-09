import os
from datetime import datetime, timedelta
from azure.storage.blob import ContainerClient, generate_account_sas, ResourceTypes, AccountSasPermissions
from pathlib import Path
from chorus.dataset.utils import *


def get_client(account_name, account_key, container_name):
    """
    Create an Azure container client

    @param account_name: Name of Storage Account
    @param account_key: Key access to the Storage Account
    @param container_name: Name of the container
    @return: A instance of a container client

    """
    try:
        sas_token = generate_account_sas(
            account_name=account_name,
            account_key=account_key,
            resource_types=ResourceTypes(service=True),
            permission=AccountSasPermissions(read=True, list=True),
            expiry=datetime.utcnow() + timedelta(hours=2)
        )
        container_client = ContainerClient(account_url='https://chorus.blob.core.windows.net',
                                           container_name=container_name,
                                           account_key=sas_token)
        return container_client
    except Exception as ex:
        print('Exception: ' + ex.__str__())
        raise ex


def get(prefix=None, name_start_with=None, ext=None, output_dir=None, parent_folder=None, download=True):
    """

    @param prefix: Prefix to the blob
    @param name_start_with: Firts part of the blob name
    @param ext: Extention of the blob
    @param output_dir: Out put dir for the downloaded files
    @param parent_folder: Parent folder for the output dir
    @param download: if file should be doawnloaded or not
    @return:
    """

    client = get_client(account_name='chorus', account_key=KEY, container_name=CONTAINER_NAME)
    blob_list = client.list_blobs(name_starts_with=prefix + name_start_with)
    file_names = []

    if download:
        base_dir = str(Path.home()) + '/.chorus'
        if not os.path.exists(base_dir):
            os.mkdir(base_dir)

        if parent_folder is not None:
            base_dir = base_dir + '/' + name_start_with
            if not os.path.exists(base_dir):
                os.mkdir(base_dir)

        if output_dir is not None:
            base_dir = base_dir + '/' + output_dir
            if not os.path.exists(base_dir):
                os.mkdir(base_dir)

    if ext is None:
        return blob_list
    else:

        for blob in blob_list:
            # print("\t" + blob.name)

            if ext == blob.name.split('.')[-1]:

                if download:
                    file_names.append(blob.name)
                    if not os.path.exists(base_dir + '/' + blob.name.split('/')[-1]):
                        try:
                            with open(file=base_dir + '/' + blob.name.split('/')[-1], mode="wb") as download_file:
                                download_file.write(client.download_blob(blob).readall())

                        except Exception as e:
                            # TODO: Manage the exception
                            pass
                else:
                    file_names.append(blob.name)
    return file_names


def get_annotation(site, type='bounding', download=True):
    if type == 'bounding':
        output = get(prefix=PREFIX_ANNOTATIONS_BOUNDING_BOX, name_start_with=site, ext='txt', output_dir='annotation',
                     parent_folder=site, download=download)
    elif type == 'presence':
        output = get(prefix=PREFIX_ANNOTATIONS_PRESENCE_ABSENCE, name_start_with=site, ext='txt',
                     output_dir='annotation', parent_folder=site, download=download)
    else:
        raise Exception('Type ' + type + ' is not available in the storage account')
    return output


def get_environmental_variables(site, type=None, download=True):
    if type == 'planetary':
        output = get(prefix=PREFIX_ENVIRONMENTAL_VARIABLES_PLANETARYCOMPUTER, name_start_with=site, ext='csv',
                     output_dir='enviromental', parent_folder=site, download=download)
    elif type == 'weather':
        output = get(prefix=PREFIX_ENVIRONMENTAL_VARIABLES_WEATHERSTATIONS, name_start_with=site, ext='csv',
                     output_dir='enviromental', parent_folder=site, download=download)
    else:
        raise Exception('Type ' + type + ' of environmental variables is not available in the storage account')
    return output


def get_dataloggers(site=None, download=True):
    if site is not None:
        output = get(prefix=PREFIX_DATALOGGERS[site], name_start_with=site, ext='xlsx', output_dir='dataloggers',
                     parent_folder=site, download=download)
    else:
        raise Exception("Site is a mandatory parameter")
    return output


def get_records(site=None, download=True):
    if site is not None:
        output = get(prefix=PREFIX_RECORDS[site], name_start_with=site, ext='wav', output_dir='records',
                     parent_folder=site, download=download)
    else:
        raise Exception("Site is a mandatory parameter")
    return output


def get_raw_data(site=None):
    if site is not None:
        get_annotation(site, type='bounding')
        get_annotation(site, type='presence')
        get_environmental_variables(site, type='planetary')
        get_environmental_variables(site, type='weather')
        get_dataloggers(site)
        get_records(site)

    else:
        raise Exception("Site is a mandatory parameter")


def get_data_by_filename(file_name_list):
    if file_name_list is None:
        raise Exception('file_name_list should a not empty list of file names')
    else:
        for i in file_name_list:
            file_split = i.split('.')
            file_name = ''.join(file_split[0:-1])
            ext = file_split[-1]
            get(prefix='/'.join(file_name.split('/')[0:-1]) + '/', name_start_with=file_name.split('/')[-1], ext=ext,
                output_dir='raw_data', download=True)
