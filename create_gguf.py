import argparse
import subprocess
import os
from huggingface_hub import create_repo, HfApi, ModelCard
import shutil

# Parse command-line arguments
parser = argparse.ArgumentParser(description='Quantize and process a model with AWQ.')
parser.add_argument('--model', required=True, type=str, help='The path to the model to be processed')
args = parser.parse_args()
model_name = args.model.split("/")[-1]
org_name = args.model.split("/")[0]

QUANTIZATION_METHODS = "q2_k, q3_k_l, q3_k_m, q3_k_s, q4_0, q4_1, q4_k_m, q4_k_s, q5_0, q5_1, q5_k_m, q5_k_s, q6_k, q8_0".replace(" ", "").split(",")
print(f"Downloading {model_name} ...")
with subprocess.Popen(['huggingface-cli', 'download', "--local-dir", args.model, args.model], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True) as proc:
    stdout, stderr = proc.communicate()
    print("STDOUT:", stdout)
    print("STDERR:", stderr)
os.mkdir(f"{org_name}/{model_name}-GGUF")
fp16 = f"{org_name}/{model_name}-GGUF/{model_name.lower()}.fp16.bin"
print(f"Converting {model_name} to {fp16} ...")
with subprocess.Popen(['python', 'llama.cpp/convert.py', args.model, '--outtype', 'f16', "--outfile", fp16], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True) as proc:
    stdout, stderr = proc.communicate()
    print("STDOUT:", stdout)
    print("STDERR:", stderr)
shutil.rmtree(f"{org_name}/{model_name}")
for method in QUANTIZATION_METHODS:
    qtype = f"{org_name}/{model_name}-GGUF/{model_name.lower()}.{method.upper()}.gguf"
    print(f"Quantizing {fp16}, {qtype} ...")
    with subprocess.Popen(["./llama.cpp/quantize", fp16, qtype, method], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True) as proc:
        stdout, stderr = proc.communicate()
        print("STDOUT:", stdout)
        print("STDERR:", stderr)

# Create model card
card = ModelCard.load(args.model)
if not hasattr(card.data, "tags"):
    card.data.tags = []
card.data.tags.append("gguf")
card.save(f'{org_name}/{model_name}-GGUF/README.md')
