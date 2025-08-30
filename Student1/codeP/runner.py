import torchvision.utils as vutils
from models import get_model
from datasets import get_dataset
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchaudio
import numpy as np
# import matplotlib.pyplot as plt
import os
from pathlib import Path
import random
from tqdm import tqdm
from transformers import (
    GPTNeoForCausalLM,
    GPT2Tokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)

def run_pipeline(model_name, dataset_name, options_name=None):
    # This will run in main.py
    if model_name == "dcgan":
        gan = get_model(model_name)
        dataloader = get_dataset(dataset_name)
        gan.train(dataloader, num_epochs=5)
        samples = gan.generate_samples(16)
        results_dir = os.path.join("RESULTS", "dcgan")
        os.makedirs(results_dir, exist_ok=True)
        save_path = os.path.join(results_dir, "dcgan_generated_samples.png")
        vutils.save_image(samples, save_path, normalize=True)
    elif model_name == "cgan":
        cgan = get_model(model_name)
        dataloader = get_dataset(dataset_name,model_name)
        cgan.train(dataloader, epochs=50, save_interval=10)
        gen_imgs, labels = cgan.generate_samples(num_samples=3, specific_class=7)
        results_dir = os.path.join("RESULTS", "cgan")
        os.makedirs(results_dir, exist_ok=True)
        save_path = os.path.join(results_dir, "cgan_generated_samples.png")
        vutils.save_image(gen_imgs, save_path, normalize=True)
    elif model_name == "cvae":
        cvae = get_model(model_name)
        dataloader = get_dataset(dataset_name,model_name)
        cvae.train(dataloader, epochs=50, save_interval=10)
        gen_imgs, labels = cvae.generate_samples(num_samples=10, specific_class=4)
        results_dir = os.path.join("RESULTS", "cvae")
        os.makedirs(results_dir, exist_ok=True)
        save_path = os.path.join(results_dir, "cvae_generated_samples.png")
        vutils.save_image(gen_imgs, save_path, normalize=True)
        cvae.generate_class_samples(save_path = os.path.join(results_dir, "cvae_class_samples.png"))
    elif model_name == "wavegan":
        LATENT_DIM = 100
        AUDIO_LENGTH = 16384
        NUM_EPOCHS = 50
        LEARNING_RATE = 0.0001
        generator, discriminator = get_model(model_name)
        dataset_dir = get_dataset(dataset_name)
        class AudioDataset(Dataset):
            """Custom dataset for loading audio files"""
            
            def __init__(self, audio_dir, sample_rate=16000, audio_length=16384):
                self.audio_dir = Path(audio_dir)
                self.sample_rate = sample_rate
                self.audio_length = audio_length
                
                # Get all audio files
                self.audio_files = []
                for ext in ['*.wav', '*.mp3', '*.flac']:
                    self.audio_files.extend(self.audio_dir.glob(f"**/{ext}"))
                
                print(f"Found {len(self.audio_files)} audio files")
                
            def __len__(self):
                return len(self.audio_files)
            
            def __getitem__(self, idx):
                audio_path = self.audio_files[idx]
                
                try:
                    # Load audio using torchaudio
                    waveform, sr = torchaudio.load(audio_path)
                    
                    # Convert to mono if stereo
                    if waveform.shape[0] > 1:
                        waveform = torch.mean(waveform, dim=0, keepdim=True)
                    
                    # Resample if necessary
                    if sr != self.sample_rate:
                        resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
                        waveform = resampler(waveform)
                    
                    # Ensure we have the right length
                    if waveform.shape[1] > self.audio_length:
                        # Random crop
                        start = random.randint(0, waveform.shape[1] - self.audio_length)
                        waveform = waveform[:, start:start + self.audio_length]
                    elif waveform.shape[1] < self.audio_length:
                        # Pad with zeros
                        pad_length = self.audio_length - waveform.shape[1]
                        waveform = F.pad(waveform, (0, pad_length))
                    
                    # Normalize to [-1, 1]
                    waveform = waveform / (torch.max(torch.abs(waveform)) + 1e-8)
                    
                    return waveform.squeeze(0)  # Remove channel dimension
                    
                except Exception as e:
                    print(f"Error loading {audio_path}: {e}")
                    # Return silence if file can't be loaded
                    return torch.zeros(self.audio_length)
        dataset = AudioDataset(dataset_dir, audio_length=16384)
        dataloader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=2)
        test_batch_size = 2
        test_noise = torch.randn(test_batch_size, LATENT_DIM).to(torch.device)
        test_audio = torch.randn(test_batch_size, AUDIO_LENGTH).to(torch.device)
        try:
            gen_output = generator(test_noise)
            print(f"Generator output shape: {gen_output.shape}")
            
            disc_output = discriminator(test_audio)
            print(f"Discriminator output shape: {disc_output.shape}")
            
            disc_gen_output = discriminator(gen_output)
            print(f"Discriminator on generated audio shape: {disc_gen_output.shape}")
            
            print("✓ Model architectures are working correctly!")
            
        except Exception as e:
            print(f"✗ Error in model architecture: {e}")
            return
        def train_wavegan(generator, discriminator, dataloader, num_epochs=100, lr=0.0001):
            """Train WaveGAN"""
            
            criterion = nn.BCEWithLogitsLoss()
            
            optimizer_G = optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))
            optimizer_D = optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))
            
            generator.train()
            discriminator.train()

            fixed_noise = torch.randn(16, generator.latent_dim).to(torch.device)

            G_losses = []
            D_losses = []
            
            for epoch in range(num_epochs):
                epoch_G_loss = 0
                epoch_D_loss = 0
                
                progress_bar = tqdm(dataloader, desc=f'Epoch {epoch+1}/{num_epochs}')
                
                for i, real_audio in enumerate(progress_bar):
                    batch_size = real_audio.size(0)
                    real_audio = real_audio.to(torch.device)
                    
                    real_labels = torch.ones(batch_size, 1).to(torch.device)
                    fake_labels = torch.zeros(batch_size, 1).to(torch.device)
                    
                    optimizer_D.zero_grad()
                    
                    output_real = discriminator(real_audio)
                    d_loss_real = criterion(output_real, real_labels)

                    noise = torch.randn(batch_size, generator.latent_dim).to(torch.device)
                    fake_audio = generator(noise)
                    output_fake = discriminator(fake_audio.detach())
                    d_loss_fake = criterion(output_fake, fake_labels)
                    
                    d_loss = d_loss_real + d_loss_fake
                    d_loss.backward()
                    optimizer_D.step()
                    
                    optimizer_G.zero_grad()
                    
                    output_fake = discriminator(fake_audio)
                    g_loss = criterion(output_fake, real_labels)
                    
                    g_loss.backward()
                    optimizer_G.step()
                    
                    epoch_G_loss += g_loss.item()
                    epoch_D_loss += d_loss.item()
                    
                    progress_bar.set_postfix({
                        'G_loss': f'{g_loss.item():.4f}',
                        'D_loss': f'{d_loss.item():.4f}'
                    })
                
                avg_G_loss = epoch_G_loss / len(dataloader)
                avg_D_loss = epoch_D_loss / len(dataloader)
                
                G_losses.append(avg_G_loss)
                D_losses.append(avg_D_loss)
                
                print(f'Epoch [{epoch+1}/{num_epochs}] - G_loss: {avg_G_loss:.4f}, D_loss: {avg_D_loss:.4f}')
                
                if (epoch + 1) % 10 == 0:
                    generator.eval()
                    with torch.no_grad():
                        fake_samples = generator(fixed_noise)
                        sample_audio = fake_samples[0].cpu().numpy()
                        
                        os.makedirs('RESULTS/wavegan', exist_ok=True)
                        
                        torchaudio.save(
                            f'RESULTS/wavegan/train_samples/sample_epoch_{epoch+1}.wav',
                            torch.tensor(sample_audio).unsqueeze(0),
                            16000
                        )
                    generator.train()
            
            return G_losses, D_losses

        def generate_samples(generator, num_samples=5):
            """Generate audio samples"""
            generator.eval()
            
            os.makedirs('final_samples', exist_ok=True)
            
            with torch.no_grad():
                for i in range(num_samples):
                    noise = torch.randn(5, generator.latent_dim).to(torch.device)
                    fake_audio = generator(noise)
                    
                    audio_np = fake_audio[0].cpu().numpy()
                    
                    torchaudio.save(
                        f'RESULTS/wavegan/final_samples/generated_sample_{i+1}.wav',
                        torch.tensor(audio_np).unsqueeze(0),
                        16000
                    )
                    
                    print(f"Generated sample {i+1} saved")
        print(f"\nStarting training for {NUM_EPOCHS} epochs...")
        train_wavegan(generator, discriminator, dataloader,num_epochs=NUM_EPOCHS, lr=LEARNING_RATE)
        print("\nGenerating final samples...")
        generate_samples(generator, num_samples=5)
    elif model_name == "gptneo":
        TRAINING_DATA_PATH = "./DATA/gptneo/training"
        TESTING_DATA_PATH = "./DATA/gptneo/testing"
        RESULTS_PATH = "./RESULTS/gptneo/"
        training_texts = get_dataset(TRAINING_DATA_PATH, "training")
        testing_texts = get_dataset(TESTING_DATA_PATH, "testing")

        MODEL_NAME = "EleutherAI/gpt-neo-1.3B"  # Smaller model for faster testing
        tokenizer = GPT2Tokenizer.from_pretrained(MODEL_NAME)
        model = GPTNeoForCausalLM.from_pretrained(MODEL_NAME)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        def split_questions(text):
            """Split text into individual questions"""
            # Split by question marks followed by whitespace or newlines
            import re

            # First, try splitting by question marks
            potential_questions = re.split(r'\?\s*\n', text)

            # Clean up and filter out empty strings
            questions = []
            for q in potential_questions:
                q = q.strip()
                if q and len(q) > 5:  # Only keep non-empty questions with reasonable length
                    # Add question mark back if it was removed
                    if not q.endswith('?'):
                        q += '?'
                    questions.append(q)

            # If no proper split happened, treat the whole text as one prompt
            if len(questions) <= 1:
                return [text.strip()]

            return questions
        def generate_response_debug(model, tokenizer, prompt, max_length=200, temperature=0.8):
            """Generate response with detailed debugging and better parameters"""
            print(f"\n--- Generating response for prompt ---")
            print(f"Prompt: {prompt}")

            try:
                # Add some context to make it clear we want a complete answer
                formatted_prompt = f"Question: {prompt}\nAnswer:"

                # Tokenize input
                inputs = tokenizer.encode(formatted_prompt, return_tensors='pt', truncation=True, max_length=512)
                print(f"Input tokens shape: {inputs.shape}")
                print(f"Formatted prompt: {formatted_prompt}")

                # Generate with better parameters
                with torch.no_grad():
                    outputs = model.generate(
                        inputs,
                        max_length=len(inputs[0]) + max_length,
                        min_length=len(inputs[0]) + 20,  # Ensure minimum response length
                        temperature=temperature,
                        top_p=0.9,
                        top_k=50,
                        do_sample=True,
                        pad_token_id=tokenizer.eos_token_id,
                        eos_token_id=tokenizer.eos_token_id,
                        no_repeat_ngram_size=3,
                        repetition_penalty=1.1,
                        length_penalty=1.0,
                        early_stopping=False  # Don't stop early
                    )

                # Decode response
                full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)

                # Extract only the generated part (after "Answer:")
                if "Answer:" in full_response:
                    generated_part = full_response.split("Answer:")[-1].strip()
                else:
                    generated_part = full_response[len(formatted_prompt):].strip()

                # If response is too short, try alternative approach
                if len(generated_part) < 10:
                    print("Response too short, trying alternative approach...")
                    simple_inputs = tokenizer.encode(prompt, return_tensors='pt', truncation=True, max_length=512)

                    with torch.no_grad():
                        outputs = model.generate(
                            simple_inputs,
                            max_length=len(simple_inputs[0]) + max_length,
                            min_length=len(simple_inputs[0]) + 30,
                            temperature=0.9,
                            top_p=0.95,
                            do_sample=True,
                            pad_token_id=tokenizer.eos_token_id,
                            no_repeat_ngram_size=2,
                            repetition_penalty=1.2
                        )

                    full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
                    generated_part = full_response[len(prompt):].strip()

                print(f"✓ Generated response ({len(generated_part)} characters)")
                print(f"Response preview: {generated_part[:100]}...")
                return generated_part

            except Exception as e:
                print(f"✗ Generation failed: {e}")
                return f"Error generating response: {e}"
        if testing_texts:
            print(f"Processing {len(testing_texts)} files from testing directory:")

            all_results = []
            question_counter = 1

            for file_idx, file_content in enumerate(testing_texts):
                print(f"\n{'='*50}")
                print(f"PROCESSING FILE {file_idx + 1}")
                print(f"{'='*50}")

                # Split the file content into individual questions
                questions = split_questions(file_content)
                print(f"Found {len(questions)} questions in this file:")

                for q_idx, question in enumerate(questions):
                    print(f"\n{'='*40}")
                    print(f"QUESTION {question_counter} (File {file_idx + 1}, Q{q_idx + 1}):")
                    print(f"{'='*40}")
                    print(f"Question: {question}")

                    response = generate_response_debug(model, tokenizer, question)

                    print(f"\nGENERATED RESPONSE:")
                    print(f"{'='*40}")
                    print(response)
                    print(f"{'='*40}")

                    # Save individual result
                    result_filename = f"question_{question_counter}_response.txt"
                    result_path = os.path.join(RESULTS_PATH, result_filename)

                    try:
                        with open(result_path, 'w', encoding='utf-8') as f:
                            f.write(f"QUESTION:\n{question}\n\n")
                            f.write(f"RESPONSE:\n{response}\n")
                        print(f"✓ Saved result to: {result_filename}")
                    except Exception as e:
                        print(f"✗ Failed to save result: {e}")

                    # Store for consolidated results
                    all_results.append({
                        'question_number': question_counter,
                        'file_index': file_idx + 1,
                        'question': question,
                        'response': response
                    })

                    question_counter += 1

            # Save consolidated results
            consolidated_filename = "all_questions_and_answers.txt"
            consolidated_path = os.path.join(RESULTS_PATH, consolidated_filename)

            try:
                with open(consolidated_path, 'w', encoding='utf-8') as f:
                    f.write("NORTH SOUTH UNIVERSITY - QUESTIONS AND ANSWERS\n")
                    f.write("=" * 60 + "\n\n")

                    for result in all_results:
                        f.write(f"QUESTION {result['question_number']}:\n")
                        f.write(f"{result['question']}\n\n")
                        f.write(f"ANSWER:\n")
                        f.write(f"{result['response']}\n\n")
                        f.write("-" * 50 + "\n\n")

                print(f"\n✓ Saved consolidated results to: {consolidated_filename}")
                print(f"✓ Total questions processed: {len(all_results)}")

            except Exception as e:
                print(f"✗ Failed to save consolidated results: {e}")

        else:
            print("No testing data available. Creating sample prompts for testing...")

            # Create sample prompts for testing
            sample_prompts = [
                "What is artificial intelligence?",
                "How does machine learning work?",
                "What are the benefits of deep learning?"
            ]

            for i, prompt in enumerate(sample_prompts):
                print(f"\n{'='*40}")
                print(f"SAMPLE PROMPT {i+1}:")
                print(f"{'='*40}")
                print(f"Prompt: {prompt}")

                response = generate_response_debug(model, tokenizer, prompt)

                print(f"\nGENERATED RESPONSE:")
                print(f"{'='*40}")
                print(response)
                print(f"{'='*40}")

                # Save to file
                result_filename = f"sample_result_{i+1}.txt"
                result_path = os.path.join(RESULTS_PATH, result_filename)

                try:
                    with open(result_path, 'w', encoding='utf-8') as f:
                        f.write(f"PROMPT:\n{prompt}\n\n")
                        f.write(f"RESPONSE:\n{response}\n")
                    print(f"✓ Saved result to: {result_filename}")
                except Exception as e:
                    print(f"✗ Failed to save result: {e}")
        if os.path.exists(RESULTS_PATH):
            result_files = os.listdir(RESULTS_PATH)
            print(f"Files created in Results folder: {result_files}")

            for file in result_files:
                if file.endswith('.txt'):
                    file_path = os.path.join(RESULTS_PATH, file)
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            content = f.read()
                            print(f"\n--- Content of {file} ---")
                            print(content[:300] + "..." if len(content) > 300 else content)
                    except Exception as e:
                        print(f"Error reading {file}: {e}")
        else:
            print("Results folder not found!")
    elif model_name == "gpt2":
        # Get dataset info
        
        dataset_info = get_dataset(dataset_name, model_name)
        
        if dataset_info is None:
            print("Failed to load dataset. Please check your training data.")
            return
        
        # Get model
        gpt2_model = get_model(model_name)
        
        # Train the model
        print("Starting GPT-2 fine-tuning...")
        model_path = gpt2_model.train(
            train_file_path=dataset_info['training_file_path'],
            block_size=dataset_info['block_size']
        )
        
        # Set up paths
        results_dir = os.path.join("RESULTS", "gpt2")
        os.makedirs(results_dir, exist_ok=True)
        
        # Process prompts from testing file
        prompt_file_path = os.path.join("DATA", "testing", "sample_prompt.txt")
        
        
        # Process prompts and generate results
        print("Processing prompts and generating results...")
        results = gpt2_model.process_prompts(prompt_file_path, results_dir)
        
        # Generate some sample outputs
        # print("Generating additional samples...")
        # samples = gpt2_model.generate_samples(num_samples=5)
        
        # Save samples
        samples_path = os.path.join(results_dir, "sample_outputs.txt")
        with open(samples_path, 'w', encoding='utf-8') as f:
            f.write("GPT-2 Fine-tuned Model Sample Outputs\n")
            f.write("="*50 + "\n\n")
            
            for i, sample in enumerate(samples, 1):
                f.write(f"SAMPLE {i}\n")
                f.write(f"Prompt: {sample['prompt']}\n")
                f.write(f"Generated on: {sample['timestamp']}\n")
                f.write("-"*30 + "\n")
                f.write("Generated Text:\n")
                f.write(sample['generated_text'])
                f.write("\n" + "="*50 + "\n\n")
        
        print(f"Sample outputs saved to: {samples_path}")
        print(f"GPT-2 fine-tuning and inference completed!")
    else:
        return
