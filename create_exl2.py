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
target_name = f"{org_name}/{model_name}-EXL2"

print(f"Downloading {model_name} ...")
with subprocess.Popen(['huggingface-cli', 'download', "--local-dir", args.model, args.model], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True) as proc:
    stdout, stderr = proc.communicate()
    print("STDOUT:", stdout)
    print("STDERR:", stderr)
os.mkdir(target_name)
os.mkdir('./exl2')
print(f"Converting {args.model} to {target_name} ...")
with subprocess.Popen(['python', 'exllamav2/convert.py', '-i', args.model, '-cf', target_name, '-o', 'exl2', '-c', 'wikitext-test.parquet', '-b', '5.0'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True) as proc:
    stdout, stderr = proc.communicate()
    print("STDOUT:", stdout)
    print("STDERR:", stderr)

shutil.rmtree(args.model)
shutil.rmtree('./exl2')
# Create model card
card = ModelCard.load(args.model)
if card.data.tags is None:
    card.data.tags = []
card.data.tags.append("exl2")
card.save(f'{target_name}/README.md')
