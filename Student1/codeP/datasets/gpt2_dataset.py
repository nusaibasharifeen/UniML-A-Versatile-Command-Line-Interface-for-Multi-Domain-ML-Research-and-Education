import os
import re
import docx
from PyPDF2 import PdfReader
from transformers import TextDataset, DataCollatorForLanguageModeling

def read_pdf(file_path):
    """Read text from PDF file"""
    with open(file_path, "rb") as file:
        pdf_reader = PdfReader(file)
        text = ""
        for page_num in range(len(pdf_reader.pages)):
            text += pdf_reader.pages[page_num].extract_text()
    return text

def read_word(file_path):
    """Read text from Word document"""
    doc = docx.Document(file_path)
    text = ""
    for paragraph in doc.paragraphs:
        text += paragraph.text + "\n"
    return text

def read_txt(file_path):
    """Read text from text file"""
    with open(file_path, "r", encoding='utf-8') as file:
        text = file.read()
    return text

def read_documents_from_directory(directory):
    """Read all supported documents from directory"""
    combined_text = ""
    supported_files = []
    
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        try:
            if filename.endswith(".pdf"):
                combined_text += read_pdf(file_path)
                supported_files.append(filename)
            elif filename.endswith(".docx"):
                combined_text += read_word(file_path)
                supported_files.append(filename)
            elif filename.endswith(".txt"):
                combined_text += read_txt(file_path)
                supported_files.append(filename)
        except Exception as e:
            print(f"Error reading {filename}: {str(e)}")
    
    print(f"Successfully read {len(supported_files)} files: {supported_files}")
    return combined_text

def create_training_file(text_data, output_path):
    """Create training file from text data"""
    # Clean up the text
    text_data = re.sub(r'\n+', '\n', text_data).strip()
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save the training data
    with open(output_path, "w", encoding='utf-8') as f:
        f.write(text_data)
    
    print(f"Training data saved to {output_path}")
    print(f"Data length: {len(text_data)} characters")
    
    return output_path

def load_data(train_directory='./DATA/training/', block_size=256):
    """Load and prepare training data for GPT-2"""
    
    # Check if directory exists and has files
    if not os.path.exists(train_directory):
        print(f"Directory {train_directory} does not exist. Please create it and add your training documents.")
        return None
    
    if not os.listdir(train_directory):
        print(f"Directory {train_directory} is empty. Please add PDF, DOCX, or TXT files for training.")
        return None
    
    # Read all documents
    text_data = read_documents_from_directory(train_directory)
    
    if not text_data.strip():
        print("No text data found in training directory.")
        return None
    
    # Create training file
    training_file_path = os.path.join(train_directory, "processed_training_data.txt")
    create_training_file(text_data, training_file_path)
    
    return {
        'training_file_path': training_file_path,
        'text_length': len(text_data),
        'block_size': block_size
    }

def load_dataset_for_training(file_path, tokenizer, block_size=256):
    """Load dataset for training"""
    dataset = TextDataset(
        tokenizer=tokenizer,
        file_path=file_path,
        block_size=block_size,
    )
    return dataset

def load_data_collator(tokenizer, mlm=False):
    """Load data collator for language modeling"""
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=mlm,
    )
    return data_collator