import glob
import os


## load_text_files_debug
def get_dataset(directory_path, file_type=""):
    """Load all txt files from a directory with detailed debugging"""
    print(f"\n=== LOADING {file_type.upper()} FILES ===")

    if not os.path.exists(directory_path):
        print(f"ERROR: Directory {directory_path} does not exist!")
        return []

    txt_files = glob.glob(os.path.join(directory_path, "*.txt"))
    print(f"Found {len(txt_files)} .txt files in {directory_path}")

    if len(txt_files) == 0:
        print("No .txt files found!")
        # Check for other file types
        all_files = os.listdir(directory_path)
        print(f"All files in directory: {all_files}")
        return []

    all_text = []

    for file_path in txt_files:
        try:
            print(f"\nReading file: {os.path.basename(file_path)}")
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read().strip()
                if content:
                    all_text.append(content)
                    print(f"✓ Loaded successfully - {len(content)} characters")
                    print(f"First 100 characters: {content[:100]}...")
                else:
                    print("✗ File is empty")
        except Exception as e:
            print(f"✗ Error loading {file_path}: {e}")

    print(f"\nTotal {file_type} texts loaded: {len(all_text)}")
    return all_text