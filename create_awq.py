from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer
import torch
import argparse

# Parse command-line arguments
parser = argparse.ArgumentParser(description='Quantize and process a model with AWQ.')
parser.add_argument('--model', type=str, help='The path to the model to be processed')
args = parser.parse_args()
model_path = args.model

quant_name =  model_path.split("/")[-1] + "-AWQ"
quant_config = { "zero_point": True, "q_group_size": 128, "w_bit": 4, "version": "GEMM" }

#Load model
model = AutoAWQForCausalLM.from_pretrained(
  model_path, safetensors=True, **{"low_cpu_mem_usage": True}
)
tokenizer = AutoTokenizer.from_pretrained(model_path,
                                          trust_remote_code = True)

# Quantize
model.quantize(
               tokenizer,
               quant_config=quant_config,
)


model.save_quantized(quant_name, safetensors=True , shard_size="10GB")
tokenizer.save_pretrained(quant_name)
