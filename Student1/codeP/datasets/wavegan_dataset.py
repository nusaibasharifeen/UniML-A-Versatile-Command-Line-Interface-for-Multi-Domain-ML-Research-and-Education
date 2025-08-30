import os

from pathlib import Path
import urllib
import zipfile


def download_dataset():
    """Download and extract the ESC-50 dataset"""
    dataset_url = "https://github.com/karolpiczak/ESC-50/archive/master.zip"
    dataset_dir = "./DATA/dcgan_dataset/training/data/esc50_dataset"
    
    if not os.path.exists(dataset_dir):
        print("Downloading ESC-50 dataset...")
        os.makedirs(dataset_dir, exist_ok=True)
        
        try:
            # Download the dataset
            urllib.request.urlretrieve(dataset_url, "esc50.zip")
            
            # Extract the dataset
            with zipfile.ZipFile("esc50.zip", 'r') as zip_ref:
                zip_ref.extractall(".")
            
            # Remove the zip file
            os.remove("esc50.zip")
            print("ESC-50 dataset downloaded and extracted successfully!")
            
        except Exception as e:
            print(f"Error downloading or extracting ESC-50 dataset: {e}")
            raise Exception("Failed to download or extract ESC-50 dataset")
    
    # Search for the audio directory
    audio_dir = None
    for root, dirs, _ in os.walk(dataset_dir):
        if "audio" in dirs:
            audio_dir = os.path.join(root, "audio")
            break
    
    if audio_dir is None or not os.path.exists(audio_dir):
        print(f"Directory structure in {dataset_dir}:")
        for root, dirs, files in os.walk(dataset_dir):
            print(f"Root: {root}, Dirs: {dirs}, Files: {len(files)}")
        raise Exception("ESC-50 audio directory not found")
    
    # Verify that audio files exist
    audio_files = list(Path(audio_dir).glob("**/*.wav"))
    if not audio_files:
        print(f"No WAV files found in {audio_dir}")
        raise Exception("No audio files found in ESC-50 dataset")
    
    print(f"Found audio directory: {audio_dir}")
    print(f"Number of audio files: {len(audio_files)}")
    
    return audio_dir