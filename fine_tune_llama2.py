# fine_tune_llama2.py
import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from peft import get_peft_model, LoraConfig, prepare_model_for_kbit_training
import bitsandbytes as bnb
import os

# Check for GPU or MPS (Metal Performance Shaders on Mac)
if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"  # For Apple Silicon Macs
else:
    device = "cpu"
    
print(f"Using device: {device}")

# Load 4-bit quantized pre-trained LLaMA-2 model and tokenizer
model_name = "meta-llama/Llama-2-7b"  # Can use smaller models like "TheBloke/Llama-2-7B-GPTQ" for 4-bit
print(f"Loading model: {model_name}")

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
tokenizer.pad_token = tokenizer.eos_token

# Configure 4-bit quantization and LoRA
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    load_in_4bit=True,  # Load model in 4-bit precision
    device_map="auto",
    quantization_config=bnb.nn.modules.LinearConfig4bit(
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4"
    ),
    torch_dtype=torch.float16,
)

# Prepare model for training
model = prepare_model_for_kbit_training(model)

# Define LoRA configuration
peft_config = LoraConfig(
    r=16,               # Rank of the update matrices
    lora_alpha=32,      # Parameter scaling factor
    lora_dropout=0.05,  # Dropout probability for LoRA layers
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "down_proj", "up_proj"],
)

# Get PEFT model
model = get_peft_model(model, peft_config)
print(f"LoRA parameter count: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

# Prepare dataset
train_dataset = [
    {"input": "Write a personalized outreach message for John Doe.", "output": "Hi John, I noticed your work in AI and would love to connect!"},
    {"input": "Write a personalized outreach message for Jane Smith.", "output": "Hi Jane, your recent project on NLP is impressive. Let's collaborate!"},
    {"input": "Write a personalized outreach message for Dr. Sarah Johnson who owns Mindful Therapy Group and recently posted about scheduling challenges.", "output": "Hi Dr. Johnson, I saw your post about scheduling challenges at Mindful Therapy Group. Our dashboard solution could help streamline your appointment booking and reduce no-shows by 30%. Would you be interested in a quick demo?"},
    {"input": "Write a personalized outreach message for Michael Williams who runs a small therapy practice and mentioned paperwork overwhelming his staff.", "output": "Hi Michael, I noticed your comment about paperwork overwhelming your staff. Our therapy practice dashboard has helped similar clinics reduce admin time by 40%. Would you have 15 minutes to see how it might work for your practice?"},
    {"input": "Write a personalized outreach message for Dr. Emily Chen who owns Serenity Counseling and tweeted about needing better client tracking.", "output": "Hi Dr. Chen, your tweet about needing better client tracking caught my eye. I've built dashboards specifically for therapy practices like Serenity Counseling that make client progress monitoring intuitive and HIPAA-compliant. Would you be interested in learning more?"},
    {"input": "Write a personalized outreach message for Robert Taylor who runs a therapy group practice and posted about revenue tracking issues.", "output": "Hi Robert, I saw your post about revenue tracking challenges. Many therapy group owners I've worked with faced similar issues until they implemented our financial dashboard. It provides real-time insights on insurance claims, payment status, and revenue per therapist. Would a quick demo be helpful?"},
    {"input": "Write a personalized outreach message for Dr. Lisa Martinez who mentioned her therapy practice is growing but struggling with data organization.", "output": "Hi Dr. Martinez, congratulations on your practice growth! I noticed you mentioned data organization challenges. Our dashboard solution helps therapy practices transition from spreadsheets to automated reporting, saving an average of 5 hours/week. Would you like to see how it works?"},
    {"input": "Write a personalized outreach message for James Wilson who runs Healing Paths Therapy and commented about needing better insurance claim tracking.", "output": "Hi James, I noticed your comment about insurance claim tracking at Healing Paths. Our dashboard provides real-time updates on claim status, flags delayed payments, and helps reduce rejection rates. Many practices see a 15% increase in successful claims. Would you be open to a brief discussion?"},
    {"input": "Write a personalized outreach message for Dr. Amanda Brown who owns Mindful Steps Therapy and posted about wanting data-driven insights for her practice.", "output": "Hi Dr. Brown, your post about wanting data-driven insights for Mindful Steps resonated with me. I've developed dashboards that help therapy practices visualize client outcomes, therapist utilization, and business growth. Would you be interested in seeing a personalized demo?"},
    {"input": "Write a personalized outreach message for Thomas Garcia who runs a family therapy practice and mentioned trouble balancing client care with business management.", "output": "Hi Thomas, I saw your comment about the challenge of balancing client care with business management. Our dashboard solution was designed specifically for therapy practice owners like you, giving you the key metrics without the time investment. Would a 15-minute call be helpful to discuss how it might work for your practice?"},
    {"input": "Write a personalized outreach message for Dr. Nicole White who owns Balanced Mind Therapy and posted about needing better insights into which referral sources are most valuable.", "output": "Hi Dr. White, I noticed your interest in tracking referral sources at Balanced Mind. Our therapy practice dashboard includes an automated referral analytics feature that has helped practices increase high-quality referrals by up to 25%. Would you like to see how it works?"},
    {"input": "Write a personalized outreach message for Kevin Harris who runs a group practice and mentioned struggling with tracking individual therapist performance.", "output": "Hi Kevin, your post about tracking therapist performance caught my attention. Our dashboard provides clear metrics on session counts, client retention, and revenue per therapist without adding any extra work for your team. Would next week work for a quick demo?"},
    {"input": "Write a personalized outreach message for Dr. Rebecca Johnson who owns Healing Hearts Counseling and tweeted about wanting to make more data-driven decisions for her practice.", "output": "Hi Dr. Johnson, I saw your tweet about making data-driven decisions at Healing Hearts. I've built custom dashboards for therapy practices that transform complex data into actionable insights without any technical expertise required. Would you be interested in seeing what's possible for your practice?"},
    {"input": "Write a personalized outreach message for Daniel Smith who runs a teletherapy practice and mentioned challenges with monitoring client engagement and outcomes.", "output": "Hi Daniel, I noticed your comment about monitoring engagement in your teletherapy practice. Our dashboard includes specific teletherapy metrics like session attendance rates, engagement scores, and outcome tracking that have helped similar practices improve retention by 20%. Would you have time for a brief conversation about how it might help your practice?"}
]

# Format dataset for instruction fine-tuning
def format_instruction(example):
    return f"### Instruction:\n{example['input']}\n\n### Response:\n{example['output']}"

formatted_data = [{"text": format_instruction(example)} for example in train_dataset]

# Convert to HuggingFace Dataset
hf_dataset = Dataset.from_dict({"text": [item["text"] for item in formatted_data]})

# Tokenize the dataset
def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=512)

tokenized_dataset = hf_dataset.map(tokenize_function, batched=True)

# Create output directory
output_dir = "models/llama2_finetuned_qlora"
os.makedirs(output_dir, exist_ok=True)

# Set up training arguments
training_args = TrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=4,       # Adjust based on your hardware
    gradient_accumulation_steps=4,       # Accumulate gradients to simulate larger batch size
    warmup_steps=10,
    max_steps=100,                        # Train for 100 steps
    learning_rate=2e-4,
    fp16=True if device != "cpu" else False,  # Use FP16 if not on CPU
    logging_steps=10,
    save_strategy="steps",
    save_steps=50,
    save_total_limit=2,                   # Keep only the last 2 saved models
    optim="paged_adamw_8bit",             # Use 8-bit Adam optimizer
    seed=42,
    data_seed=42,
)

# Create data collator
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=data_collator,
)

# Train model
print("Starting QLoRA fine-tuning...")
trainer.train()

# Save the fine-tuned model
print(f"Saving model to {output_dir}")
trainer.save_model(output_dir)
tokenizer.save_pretrained(output_dir)

# Add instructions to gitignore
with open(".gitignore", "a") as f:
    f.write("\n# Ignore model files\nmodels/\n")

print("Fine-tuning complete! Model saved to 'models/llama2_finetuned_qlora'.")
print("This QLoRA approach only trained ~0.1% of the full model parameters.")