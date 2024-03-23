import argparse
import torch
import random
import sys
from transformers import AutoModelForCausalLM,AutoTokenizer
from datasets import load_dataset
import torch

# Parse command-line arguments
parser = argparse.ArgumentParser(description='Quantize and process a model with AWQ.')
parser.add_argument('model', type=str, help='The path to the model to be processed')
args = parser.parse_args()
model_name = args.model
save_folder = model_name + "-bf16"

print(f"Loading model {model_name} ...\n")
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
model.to("cuda", dtype=torch.bfloat16)

print(f"Saving quantized model {save_folder} ...\n")
model.save_pretrained(save_folder, use_safetensors=True)
tokenizer.save_pretrained(save_folder)
