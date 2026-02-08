"""
Download 3D meshes from Omni6DPose dataset
Run this script from the meshup_v2 directory
"""

import json
import os
import requests
from tqdm import tqdm

# Configuration
DATA_PATH = 'data/Omni6DPose/'  # Where meshes will be saved
PROXY = None  # Set to 'http://proxy-server:port' if behind firewall

# Setup proxy
proxies = {'http': PROXY, 'https': PROXY} if PROXY else None

# Download the data_links.json file first
DATA_LINKS_URL = 'https://raw.githubusercontent.com/Omni6DPose/Omni6DPoseAPI/main/data_links.json'

print("Downloading data links configuration...")
response = requests.get(DATA_LINKS_URL, proxies=proxies)
data = response.json()

# Create directory
os.makedirs(DATA_PATH + 'PAM', exist_ok=True)


def download_with_retries(url, proxies=None, headers={}, timeout=30, max_retries=5, retry_delay=2):
    """Download with retry mechanism"""
    from time import sleep
    for attempt in range(max_retries):
        try:
            response = requests.get(url, headers=headers, stream=True, proxies=proxies, timeout=timeout)
            response.raise_for_status()
            return response
        except (requests.Timeout, requests.ConnectionError):
            print(f"Request timed out or connection error, retry {attempt + 1}/{max_retries}...")
            sleep(retry_delay)
        except requests.HTTPError as err:
            print(f"HTTP error: {err}")
            return None
    print("All retry attempts failed")
    return None


def download_file(url, filename):
    """Download a file with progress bar and resume capability"""
    progress_bar = None
    try:
        hidden_marker = filename.replace(filename.split('/')[-1], '.' + filename.split('/')[-1] + '.done')
        
        if os.path.exists(hidden_marker):
            print(f"{filename} already exists!")
            return True
        elif os.path.exists(filename):
            print(f"\033[93m{filename} already exists but incomplete! Resuming download...\033[0m")
            resume_byte_pos = os.path.getsize(filename)
        else:
            resume_byte_pos = 0
            print(f"\033[93mDownloading {filename}...\033[0m")

        headers = {'Range': f'bytes={resume_byte_pos}-'} if resume_byte_pos else {}
        url = url.replace('dl=0', 'dl=1')
        response = download_with_retries(url, headers=headers, proxies=proxies)
        
        if not response:
            return False
            
        total_size_in_bytes = int(response.headers.get('content-length', 0))
        progress_bar = tqdm(total=total_size_in_bytes, unit='iB', unit_scale=True)
        
        if response.status_code in (200, 206):
            mode = 'ab' if response.status_code == 206 else 'wb'
            with open(filename, mode) as f:
                for chunk in response.iter_content(chunk_size=8192):
                    progress_bar.update(len(chunk))
                    f.write(chunk)
            progress_bar.close()
            with open(hidden_marker, 'w') as f:
                f.write('done')
            return True
        else:
            print(f'\033[91mFailed to download {filename}! Status code {response.status_code}\033[0m')
            return False
    
    except (KeyboardInterrupt, requests.RequestException, Exception) as e:
        print(f'\033[91mError downloading {filename}! {e}\033[0m')
        return False
    
    finally:
        if progress_bar is not None:
            progress_bar.close()


# Download PAM dataset (contains the 3D meshes)
print("\n" + "="*80)
print("DOWNLOADING PAM DATASET (3D Meshes)")
print("="*80)
print("The PAM dataset contains pose-aligned 3D models.")
print("Total size: ~14GB")
print("="*80 + "\n")

PAM_links = data['PAM']
fail_list = []

for item, link in PAM_links.items():
    print(f'\033[93mDownloading {item} ...\033[0m')
    res = download_file(link, f"{DATA_PATH}PAM/{item}")
    if not res:
        fail_list.append(item)

if len(fail_list) == 0:
    print('\033[92mPAM data downloaded successfully!\033[0m')
    print(f"\nMeshes are saved in: {DATA_PATH}PAM/")
    print("\nNext step: Unzip the downloaded file(s)")
else:
    print('\n' + '-'*100)
    print('\033[91mPAM data download failed for:\033[0m')
    print(fail_list)
    print('\033[91mPlease rerun the script to retry.\033[0m')


# Unzip instructions
print("\n" + "="*80)
print("TO UNZIP THE MESHES:")
print("="*80)
print("Run the following command:")
print(f"  cd {DATA_PATH}PAM/")
print(f"  unzip '*.zip'")
print("="*80)
