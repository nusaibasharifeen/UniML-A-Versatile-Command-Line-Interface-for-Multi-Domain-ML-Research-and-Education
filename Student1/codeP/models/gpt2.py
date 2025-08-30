# --- BEGIN namespace shim ---------------------------------
import sys, importlib
if 'datasets' in sys.modules and not hasattr(sys.modules['datasets'], 'Dataset'):
    # Our local image-dataset package is hijacking the name.
    sys.modules['local_datasets'] = sys.modules.pop('datasets')
importlib.import_module('datasets')   # now loads Hugging Face package
# --- END namespace shim -----------------------------------

import torch
import os
from datetime import datetime
from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments
from datasets.gpt2_dataset import load_dataset_for_training, load_data_collator

class GPT2FineTuner:
    """
    Complete GPT-2 Fine-tuning implementation
    """
    
    def __init__(self, model_name='gpt2', output_dir='./DATA/output', 
                 per_device_train_batch_size=2, num_train_epochs=30.0, 
                 save_steps=500, max_length=100):
        
        self.model_name = model_name
        self.output_dir = output_dir
        self.per_device_train_batch_size = per_device_train_batch_size
        self.num_train_epochs = num_train_epochs
        self.save_steps = save_steps
        self.max_length = max_length
        
        # Device setup
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Initialize tokenizer and model
        self.tokenizer = None
        self.model = None
        
    def train(self, train_file_path, block_size=256):
        """Train the GPT-2 model"""
        
        print("Loading tokenizer and model...")
        self.tokenizer = GPT2Tokenizer.from_pretrained(self.model_name)
        
        # Add padding token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load dataset and data collator
        train_dataset = load_dataset_for_training(train_file_path, self.tokenizer, block_size)
        data_collator = load_data_collator(self.tokenizer)
        
        # Save tokenizer
        self.tokenizer.save_pretrained(self.output_dir)
        
        # Load and save model
        self.model = GPT2LMHeadModel.from_pretrained(self.model_name)
        self.model.save_pretrained(self.output_dir)
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=self.output_dir,
            overwrite_output_dir=True,
            per_device_train_batch_size=self.per_device_train_batch_size,
            num_train_epochs=self.num_train_epochs,
            save_steps=self.save_steps,
            logging_steps=100,
            save_total_limit=2,
            prediction_loss_only=True,
            remove_unused_columns=False,
        )
        
        # Create trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            data_collator=data_collator,
            train_dataset=train_dataset,
        )
        
        print("Starting training...")
        trainer.train()
        trainer.save_model()
        print("Training completed!")
        
        return self.output_dir
    
    def load_trained_model(self, model_path=None):
        """Load trained model and tokenizer"""
        if model_path is None:
            model_path = self.output_dir
            
        self.model = GPT2LMHeadModel.from_pretrained(model_path)
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_path)
        self.model.to(self.device)
        
        return self.model, self.tokenizer
    
    def generate_text(self, sequence, max_length=None):
        """Generate text from a given sequence"""
        if max_length is None:
            max_length = self.max_length
            
        if self.model is None or self.tokenizer is None:
            self.load_trained_model()
        
        # Tokenize input
        ids = self.tokenizer.encode(f'{sequence}', return_tensors='pt').to(self.device)
        
        # Generate text
        with torch.no_grad():
            final_outputs = self.model.generate(
                ids,
                do_sample=True,
                max_length=max_length,
                pad_token_id=self.model.config.eos_token_id,
                top_k=50,
                top_p=0.95,
                temperature=0.8,
            )
        
        generated_text = self.tokenizer.decode(final_outputs[0], skip_special_tokens=True)
        return generated_text
    
    def read_prompts_from_file(self, file_path):
        """Read prompts from a text file"""
        prompts = []
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                for line_num, line in enumerate(file, 1):
                    line = line.strip()
                    # Skip empty lines and comments
                    if line and not line.startswith('#'):
                        prompts.append({
                            'prompt': line,
                            'line_number': line_num
                        })
            return prompts
        except FileNotFoundError:
            print(f"Error: File {file_path} not found.")
            return []
        except Exception as e:
            print(f"Error reading file {file_path}: {str(e)}")
            return []
    
    def process_prompts(self, prompt_file_path, results_dir):
        """Process multiple prompts from file and save results"""
        
        # Check if prompt file exists
        if not os.path.exists(prompt_file_path):
            print(f"Error: Prompt file {prompt_file_path} not found!")
            return []
        
        # Read prompts from file
        prompts = self.read_prompts_from_file(prompt_file_path)
        
        if not prompts:
            print("No valid prompts found in the file.")
            return []
        
        print(f"Found {len(prompts)} prompts to process.")
        
        # Load model if not already loaded
        if self.model is None or self.tokenizer is None:
            self.load_trained_model()
        
        # Process each prompt
        results = []
        for i, prompt_data in enumerate(prompts, 1):
            prompt = prompt_data['prompt']
            line_num = prompt_data['line_number']
            
            print(f"\nProcessing prompt {i}/{len(prompts)} (line {line_num}): {prompt[:50]}...")
            
            try:
                generated_text = self.generate_text(prompt, self.max_length)
                
                result = {
                    'prompt': prompt,
                    'generated_text': generated_text,
                    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'max_length': self.max_length,
                    'line_number': line_num
                }
                
                results.append(result)
                print(f"✓ Completed prompt {i}")
                
            except Exception as e:
                print(f"✗ Error processing prompt {i}: {str(e)}")
                continue
        
        # Save results
        if results:
            self.save_results(results, results_dir)
            print(f"\n✓ Successfully processed {len(results)} prompts!")
            print(f"Results saved in {results_dir}")
        
        return results
    
    def save_results(self, results, output_dir):
        """Save results to both individual files and a combined file"""
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save individual results
        for i, result in enumerate(results, 1):
            filename = f"result_{i:03d}.txt"
            filepath = os.path.join(output_dir, filename)
            
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(f"Prompt: {result['prompt']}\n")
                f.write(f"Generated on: {result['timestamp']}\n")
                f.write(f"Max length: {result['max_length']}\n")
                f.write("="*50 + "\n")
                f.write("Generated Answer:\n")
                f.write(result['generated_text'])
                f.write("\n")
        
        # Save combined results
        combined_filename = f"all_results_{timestamp}.txt"
        combined_filepath = os.path.join(output_dir, combined_filename)
        
        with open(combined_filepath, 'w', encoding='utf-8') as f:
            f.write(f"GPT-2 Fine-tuned Model Results\n")
            f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total prompts processed: {len(results)}\n")
            f.write("="*70 + "\n\n")
            
            for i, result in enumerate(results, 1):
                f.write(f"RESULT {i:03d}\n")
                f.write(f"Prompt: {result['prompt']}\n")
                f.write(f"Generated on: {result['timestamp']}\n")
                f.write(f"Max length: {result['max_length']}\n")
                f.write("-"*50 + "\n")
                f.write("Generated Answer:\n")
                f.write(result['generated_text'])
                f.write("\n" + "="*70 + "\n\n")
        
        print(f"Results saved to {output_dir}")
        print(f"Individual files: result_001.txt to result_{len(results):03d}.txt")
        print(f"Combined file: {combined_filename}")
        
        return combined_filepath
    
    
  