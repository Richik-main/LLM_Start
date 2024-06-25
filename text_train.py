#%%
import pandas as pd

# Define the data
data = {
    "Category": ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"],
    "Value1": [150, 300, 100, 250, 200, 350, 400, 450, 500, 600],
    "Value2": [200, 400, 150, 300, 250, 450, 500, 600, 700, 800],
    "Sum": [350, 700, 250, 550, 450, 800, 900, 1050, 1200, 1400],
    "Percentage": ["35%", "70%", "25%", "55%", "45%", "80%", "90%", "105%", "120%", "140%"],
    "Ratio": [1.33, 1.33, 1.5, 1.2, 1.25, 1.29, 1.25, 1.33, 1.4, 1.33]
}

# Convert to DataFrame
df = pd.DataFrame(data)

# Convert DataFrame to text
text_data = df.to_string(index=False)

# Save the text data to a file
with open('text_data.txt', 'w') as f:
    f.write(text_data)


#%%
from transformers import GPT2Tokenizer, GPT2LMHeadModel, TextDataset, DataCollatorForLanguageModeling, Trainer, TrainingArguments, AdamW, get_linear_schedule_with_warmup

# Load the GPT-2 tokenizer and model
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained('gpt2')

# Prepare the dataset
def load_dataset(file_path, tokenizer, block_size=128):
    return TextDataset(
        tokenizer=tokenizer,
        file_path=file_path,  # file path of the text data
        block_size=block_size)

# Define data collator
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,
)

# Define training arguments
training_args = TrainingArguments(
    output_dir='./results',
    overwrite_output_dir=True,
    num_train_epochs=100,  # Increased number of epochs
    per_device_train_batch_size=8,  # Increased batch size
    save_steps=1000,  # Save checkpoint every 1000 steps
    save_total_limit=2,
    learning_rate=5e-7,  # Adjust learning rate
    weight_decay=0.01,  # Apply weight decay
    max_grad_norm=1.0,  # Gradient clipping
    warmup_steps=500,  # Number of warmup steps
    evaluation_strategy="steps",
    eval_steps=1000,
)

# Load the dataset
train_dataset = load_dataset('text_data.txt', tokenizer)

# Create the Trainer with a learning rate scheduler
optimizer = AdamW(model.parameters(), lr=5e-7)
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=500,
    num_training_steps=len(train_dataset) // training_args.per_device_train_batch_size * training_args.num_train_epochs,
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_dataset,
    optimizers=(optimizer, scheduler),
)

# Train the model
trainer.train()


model.save_pretrained('./results')
tokenizer.save_pretrained('./results')
# %%

import torch
# Load the fine-tuned model and tokenizer
model = GPT2LMHeadModel.from_pretrained('./results')
tokenizer = GPT2Tokenizer.from_pretrained('./results')

# Generate text
input_text = "Category: A, Value1: 150, Value2: 200"
input_ids = tokenizer(input_text, return_tensors="pt").input_ids
attention_mask = torch.ones_like(input_ids)  # Assuming all tokens are attended to
output = model.generate(
    input_ids,
    attention_mask=attention_mask,
    max_length=50,
    num_return_sequences=1,
    pad_token_id=tokenizer.eos_token_id
)
# Decode the generated text
generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
print(generated_text)

# %%
