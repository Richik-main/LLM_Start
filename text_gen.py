#%%
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
#%%
def main():
    model_name = 'gpt2'
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name)
    
    model.eval()

    # Check if CUDA is available and move the model to GPU if possible
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # Function to generate text
    def generate_text(prompt, max_length=100):
        inputs = tokenizer.encode(prompt, return_tensors='pt')
        inputs = inputs.to(device)

        # Generate text
        outputs = model.generate(
            inputs, 
            max_length=max_length, 
            num_return_sequences=1,
            no_repeat_ngram_size=2,
            top_k=50,
            top_p=0.80,
            temperature=0.2,
            do_sample=True
        )
        # Decode and return the generated text
        text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return text

    # Example usage
    prompt = "Once upon a time"
    generated_text = generate_text(prompt)
    print("Generated text:")
    print(generated_text)
    # Example usage for few-shot learning
    few_shot_prompt = (
        "Example 1: \n"
        "Input: What is the capital of France?\n"
        "Output: The capital of France is Paris.\n\n"
        "Example 2: \n"
        "Input: Who wrote 'To Kill a Mockingbird'?\n"
        "Output: 'To Kill a Mockingbird' was written by Harper Lee.\n\n"
        "Example 3: \n"
        "Input: What is the capital of India?\n"
        "Output: "
    )

    generated_text = generate_text(few_shot_prompt)
    print("Generated text for few-shot inference:")
    print(generated_text)

if __name__ == "__main__":
    main()
# %%
    
##########  FLAN T5 for question answers ###########
    


from transformers import T5Tokenizer, T5ForConditionalGeneration

def main():
    model_name = 'google/flan-t5-small'  
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(model_name)
    
    
    model.eval()

    # Check if CUDA is available and move the model to GPU if possible
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    def generate_text_few_shot(prompt, max_length=150):
        inputs = tokenizer(prompt, return_tensors='pt')
        inputs = inputs.to(device)


        outputs = model.generate(
            inputs.input_ids, 
            max_length=max_length, 
            num_return_sequences=1,
            no_repeat_ngram_size=2,
            top_k=50,
            top_p=0.95,
            temperature=0.2,
            do_sample=True
        )

        # Decode and return the generated text
        text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return text

  
    few_shot_prompt = (
        "Example 1: \n"
        "Input: What is the capital of France?\n"
        "Output: The capital of France is Paris.\n\n"
        "Example 2: \n"
        "Input: Who wrote 'To Kill a Mockingbird'?\n"
        "Output: 'To Kill a Mockingbird' was written by Harper Lee.\n\n"
        "Example 3: \n"
        "Input: What is the tallest mountain in the world?\n"
        "Output: The tallest mountain in the world is "
    )

    generated_text = generate_text_few_shot(few_shot_prompt)
    print("Generated text for few-shot inference:")
    print(generated_text)

if __name__ == "__main__":
    main()
# %%
# using LLaMa for text generation #
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def main():
    model_name = "huggyllama/llama-7b" 
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

   
    def generate_text(prompt, max_length=50):
        inputs = tokenizer(prompt, return_tensors="pt")
        inputs = {key: value.to(device) for key, value in inputs.items()}

    
        outputs = model.generate(
            **inputs,
            max_length=max_length,
            num_return_sequences=1,
            no_repeat_ngram_size=2,
            top_k=50,
            top_p=0.80,
            temperature=0.7,
            do_sample=True
        )
        # Decode and return the generated text
        text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return text

    # Example usage
    prompt = "Once upon a time"
    generated_text = generate_text(prompt)
    print("Generated text:")
    print(generated_text)
    
    # Example usage for few-shot learning
    few_shot_prompt = (
        "Example 1: \n"
        "Input: What is the capital of France?\n"
        "Output: The capital of France is Paris.\n\n"
        "Example 2: \n"
        "Input: Who wrote 'To Kill a Mockingbird'?\n"
        "Output: 'To Kill a Mockingbird' was written by Harper Lee.\n\n"
        "Example 3: \n"
        "Input: What is the capital of India?\n"
        "Output: "
    )
    
    generated_text = generate_text(few_shot_prompt)
    print("Generated text for few-shot inference:")
    print(generated_text)

if __name__ == "__main__":
    main()



# %%
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def main():
    model_name = "distilgpt2" 
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    def generate_text(prompt, max_length=150):
        inputs = tokenizer(prompt, return_tensors="pt")
        inputs = {key: value.to(device) for key, value in inputs.items()}

        outputs = model.generate(
            **inputs,
            max_length=max_length,
            num_return_sequences=1,
            no_repeat_ngram_size=2,
            top_k=50,
            top_p=0.80,
            temperature=0.7,
            do_sample=True
        )
        # Decode and return the generated text
        text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return text

    # Example usage
    prompt = "Once upon a time"
    generated_text = generate_text(prompt)
    print("Generated text:")
    print(generated_text)

    # Example usage for few-shot learning
    few_shot_prompt = (
        "Example 1: \n"
        "Input: What is the capital of France?\n"
        "Output: The capital of France is Paris.\n\n"
        "Example 2: \n"
        "Input: Who wrote 'To Kill a Mockingbird'?\n"
        "Output: 'To Kill a Mockingbird' was written by Harper Lee.\n\n"
        "Example 3: \n"
        "Input: What is the capital of India?\n"
        "Output: "
    )

    generated_text = generate_text(few_shot_prompt)
    print("Generated text for few-shot inference:")
    print(generated_text)

if __name__ == "__main__":
    main()

# %%
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def main():
    model_name = "huggyllama/llama-7b"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    model.to(device)

    def generate_text(prompt, max_new_tokens=50):
        inputs = tokenizer(prompt, return_tensors="pt")
        inputs = {key: value.to(device) for key, value in inputs.items()}

        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            num_return_sequences=1,
            no_repeat_ngram_size=2,
            top_k=50,
            top_p=0.80,
            temperature=0.7,
            do_sample=True
        )
        text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return text

    # Example usage
    prompt = "Once upon a time"
    generated_text = generate_text(prompt)
    print("Generated text:")
    print(generated_text)

if __name__ == "__main__":
    main()


# %%
import torch

# Check if MPS is available
if torch.backends.mps.is_available():
    print("MPS backend is available.")

    # Check if MPS is currently in use
    if torch.backends.mps.is_built():
        print("MPS backend is built and can be used.")
        # Try allocating a tensor on the MPS device
        try:
            x = torch.tensor([1.0, 2.0, 3.0], device='mps')
            print("Tensor successfully allocated on MPS device.")
            print(x)
        except Exception as e:
            print(f"An error occurred: {e}")
    else:
        print("MPS backend is not built.")
else:
    print("MPS backend is not available.")


## This check shows that Metal Performace Shaders are up and running in my system
# %%
    
# Quantization method 
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset

def main():
    model_name = "huggyllama/llama-7b"  
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    # Apply dynamic quantization for efficiency
    model = torch.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)

    # Use MPS (Apple Silicon) if available, else use CPU
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    model.to(device)

    def generate_text(prompt, max_new_tokens=250):
        inputs = tokenizer(prompt, return_tensors="pt")
        inputs = {key: value.to(device) for key, value in inputs.items()}

        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            num_return_sequences=1,
            no_repeat_ngram_size=2,
            top_k=50,
            top_p=0.80,
            temperature=0.7,
            do_sample=True
        )
        text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return text

    # Load the dataset from Hugging Face
    dataset = load_dataset('squad', split='validation[:10]')  

    # Generate responses for each example in the dataset
    for example in dataset:
        context = example['context']
        question = example['question']
        prompt = f"Context: {context}\n\nQuestion: {question}\nAnswer:"
        generated_text = generate_text(prompt)
        print("Generated text:")
        print(generated_text)
        print("\n" + "="*50 + "\n")

if __name__ == "__main__":
    main()


# %%

# %%
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from datasets import load_dataset

def main():
    model_name = 'gpt2'
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name)
    
    model.eval()

    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # Define a soft prompt (as if it were learned for PEFT)
    soft_prompt = "You are an expert assistant. Answer the following queries with detailed and informative responses.\n\n"

    # Function to generate text with PEFT using soft prompting
    def generate_text(prompt, max_length=250):
        # Combine soft prompt with user prompt
        combined_prompt = soft_prompt + prompt
        inputs = tokenizer.encode(combined_prompt, return_tensors='pt')
        inputs = inputs.to(device)

        # Generate text
        outputs = model.generate(
            inputs, 
            max_length=max_length, 
            num_return_sequences=1,
            no_repeat_ngram_size=2,
            top_k=50,
            top_p=0.80,
            temperature=0.2,
            do_sample=True
        )
        # Decode and return the generated text
        text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return text

    # Load the dataset from Hugging Face
    dataset = load_dataset('squad', split='validation[:3]')  
    # Generate responses for each example in the dataset
    for example in dataset:
        context = example['context']
        question = example['question']
        prompt = f"Context: {context}\n\nQuestion: {question}\nAnswer:"
        generated_text = generate_text(prompt)
        print("Generated text:")
        print(generated_text)
        print("\n" + "="*50 + "\n")

if __name__ == "__main__":
    main()


