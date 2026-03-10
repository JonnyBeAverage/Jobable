from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM
from pathlib import Path
import torch

# path to your saved model
MODEL_PATH = Path(__file__).resolve().parents[1] / "model_weights" / "cover_letter_model"

# define device FIRST
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

print("Loading model from:", MODEL_PATH)
print("Using device:", device)

# load tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    str(MODEL_PATH),
    local_files_only=True
)

# load model
model = AutoModelForCausalLM.from_pretrained(
    str(MODEL_PATH),
    dtype=torch.float16 if device.type == "mps" else torch.float32,
    local_files_only=True
)

# move model to device
model.to(device)

# fix padding
tokenizer.pad_token = tokenizer.eos_token
model.config.pad_token_id = tokenizer.eos_token_id
