from transformers import AutoModelForCausalLM, AutoTokenizer
from tokenizers import AddedToken
import torch
import argparse

# Set up argument parser
parser = argparse.ArgumentParser(description='Load a model with the specified name.')
parser.add_argument('--model', type=str, help='Model name', required=True)
args = parser.parse_args()

model_name = args.model

# Load model and tokenizer with the specified model name
model = AutoModelForCausalLM.from_pretrained(model_name).half()
tokenizer = AutoTokenizer.from_pretrained(model_name)

tokenizer.add_special_tokens({"eos_token": "<|im_end|>"})

tokenizer.add_tokens(
	AddedToken("<|im_start|>",normalized=False, rstrip=True, lstrip=False)
)

# https://huggingface.co/docs/transformers/main/chat_templating
tokenizer.chat_template = "{% if not add_generation_prompt is defined %}{% set add_generation_prompt = false %}{% endif %}{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"

tokenizer.bos_token = "<s>"
tokenizer.eos_token = "<|im_end|>"
tokenizer.pad_token = "</s>"
model.eos_token = 32000

model.resize_token_embeddings(len(tokenizer))
model.save_pretrained(save_directory="./model", torch_dtype=torch.bfloat16)
tokenizer.save_pretrained(save_directory="./model", torch_dtype=torch.bfloat16)



