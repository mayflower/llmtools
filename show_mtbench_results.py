import argparse
import json
from pathlib import Path
from huggingface_hub import notebook_login
from transformers import AutoTokenizer
import time
from vllm import LLM, SamplingParams
from tqdm import tqdm

# Argument parser setup
parser = argparse.ArgumentParser(description='Process model name.')
parser.add_argument('--model', type=str, help='Model name or path', required=True)

# Parse arguments
args = parser.parse_args()

MODEL_NAME = args.model

print("Loading ", MODEL_NAME)

llm = LLM(model=MODEL_NAME, max_model_len=8192, max_num_seqs=2048, enforce_eager=True)
llm.llm_engine.tokenizer.eos_token_id = 32000

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

sampling_params = SamplingParams(temperature=0.0, max_tokens=4096, skip_special_tokens=False, stop_token_ids=[2,32000])

def create_turn(text: str) -> str:
   messages = [
   {
       "role": "system",
       "content": "Du bist ein hilfsbereiter Assistent."
   },
   {
       "role": "user",
       "content": text
   }
   ]
   return tokenizer.apply_chat_template(
       messages, tokenize=False, add_generation_prompt=True
   )


print("Evaluating ", MODEL_NAME)
start = time.time()
# Load texts from JSON file
questions_file_path = './show_mtbench_results.json'

convs = []

with open(questions_file_path, 'r', encoding='utf-8') as file:
    data = json.load(file)
    for item in tqdm(data.values()):
        turn1 = create_turn(item["turns"][0])
        output = llm.generate(turn1, sampling_params)
        result = output[0].outputs[0].text
        turn2 = create_turn(item["turns"][1])
        output = llm.generate(f"{turn1}\n{result}\n{turn2}", sampling_params)
        print(f"{output[0].outputs[0].text}\n")
        conv = f"{output[0].prompt}\n{output[0].outputs[0].text}\n"
        convs.append(conv)
        with open('convs.json', 'w', encoding='utf-8') as file2:
                json.dump(convs, file2, ensure_ascii=False, indent=4)


duration_vllm = time.time() - start

print("Duration: ", duration_vllm)

