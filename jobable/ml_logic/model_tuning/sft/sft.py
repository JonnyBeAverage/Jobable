from datasets import load_dataset
from transformers import T5Tokenizer, T5ForConditionalGeneration, Trainer, TrainingArguments
import torch
import os


dataset = load_dataset("ShashiVish/cover-letter-dataset")

def preprocess(row):
    X = f'''
    Job title: {row['Job Title']} ;
    Qualifications: {row['Preferred Qualifications']} ;
    Hiring Company: {row['Hiring Company']} ;
    Applicant Name: {row['Applicant Name']} ;
    Past Working Experience: {row['Past Working Experience']} ;
    Current Working Experience: {row['Current Working Experience']} ;
    Skillsets: {row['Skillsets']} ;
    Qualifications: {row['Qualifications']} ;
    '''

    y = row['Cover Letter']

    return {'input_text': X, 'target_text': y}

dataset = dataset.map(preprocess)

tokenizer = T5Tokenizer.from_pretrained('t5-small')

def tokenize(row):
    model_inputs = tokenizer(
        row['input_text'],
        max_length=512,
        truncation=True,
        padding='max_length'
        )

    labels = tokenizer(
        row['target_text'],
        max_length=512,
        truncation=True,
        padding='max_length'
        )

    model_inputs['labels'] = labels['input_ids']

    return model_inputs

dataset = dataset.map(tokenize, batched=True)

model = T5ForConditionalGeneration.from_pretrained('t5-small')

os.environ["WANDB_DISABLED"] = "true"

# os.makedirs("./cover_letter_model", exist_ok=True)
training_args = TrainingArguments(
    output_dir="./cover_letter_model",
    eval_strategy="epoch",  # Updated deprecated argument
    save_strategy="epoch",
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=3,
    weight_decay=0.01,
    save_total_limit=2,
    report_to=["none"]  # Disabling Weights & Biases properly
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    processing_class=tokenizer,
)


trainer.train()

model.save_pretrained("./cover_letter_model_3")
tokenizer.save_pretrained("./cover_letter_model_3")
