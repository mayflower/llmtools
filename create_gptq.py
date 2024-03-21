import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer, GPTQConfig
import torch

# Parse command-line arguments
parser = argparse.ArgumentParser(description='Quantize and process a model with AWQ.')
parser.add_argument('model', type=str, help='The path to the model to be processed')
args = parser.parse_args()
model_name = args.model

tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
quantization_config = GPTQConfig(bits=4, dataset = "c4", tokenizer=tokenizer)
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", quantization_config=quantization_config)


quantizer = GPTQQuantizer(bits=4, dataset="c4", block_name_to_quantize = "model.decoder.layers", model_seqlen = 2048)
quantized_model = quantizer.quantize_model(model, tokenizer)

save_folder = model_name + "-GPTQ"

model.save(save_folder)
tokenizer.save(save_folder)
