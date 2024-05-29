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

    def generate_text_few_shot(prompt, max_length=100):
        inputs = tokenizer(prompt, return_tensors='pt')
        inputs = inputs.to(device)


        outputs = model.generate(
            inputs.input_ids, 
            max_length=max_length, 
            num_return_sequences=1,
            no_repeat_ngram_size=2,
            top_k=50,
            top_p=0.95,
            temperature=0.7,
            do_sample=True
        )

        # Decode and return the generated text
        text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return text

    # Example usage for few-shot learning
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