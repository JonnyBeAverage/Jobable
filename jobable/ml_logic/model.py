from transformers import pipeline, AutoTokenizer

MODEL_NAME = "google/flan-t5-large"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

generator = pipeline(
    "text2text-generation",
    model=MODEL_NAME
)
