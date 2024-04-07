import argparse
import subprocess
import os
from huggingface_hub import create_repo, HfApi, ModelCard
import shutil
import torch
from hqq.engine.hf import HQQModelForCausalLM, AutoTokenizer
from hqq.models.hf.base import AutoHQQHFModel
from hqq.core.quantize import *

# Parse command-line arguments
parser = argparse.ArgumentParser(description='Quantize and process a model with HQQ.')
parser.add_argument('--model', required=True, type=str, help='The path to the model to be processed')
args = parser.parse_args()
model_name = args.model.split("/")[-1]
org_name = args.model.split("/")[0]
target_name = f"{org_name}/{model_name}-HQQ"

print(f"Downloading {model_name} ...")
with subprocess.Popen(['huggingface-cli', 'download', "--local-dir", args.model, args.model], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True) as proc:
    stdout, stderr = proc.communicate()
    print("STDOUT:", stdout)
    print("STDERR:", stderr)

os.mkdir(target_name)
print(f"Converting {args.model} to {target_name} ...")
# Quant config
quant_config = BaseQuantizeConfig(
    nbits=2,
    group_size=64
)

# Quantize model
model = HQQModelForCausalLM.from_pretrained(
    './'+args.model,
    cache_dir=".",
    attn_implementation="flash_attention_2",
    torch_dtype=torch.float16
)
tokenizer = AutoTokenizer.from_pretrained(args.model)
model.quantize_model(quant_config=quant_config, device='cuda')
model.save_quantized(target_name)
tokenizer.save_pretrained(target_name)
shutil.rmtree(args.model)
# Create model card
card = ModelCard.load(args.model)
if card.data.tags is None:
    card.data.tags = []
card.data.tags.append("hqq")
card.save(f'{target_name}/README.md')
