import argparse
import torch
import random
import sys
from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig
from datasets import load_dataset
import torch
from transformers import AutoTokenizer
import logging

logging.basicConfig(
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s", level=logging.INFO, datefmt="%Y-%m-%d %H:%M:%S"
)



# Parse command-line arguments
parser = argparse.ArgumentParser(description='Quantize and process a model with AWQ.')
parser.add_argument('model', type=str, help='The path to the model to be processed')
args = parser.parse_args()
model_name = args.model
save_folder = model_name + "-GPTQ"

quantize_config = BaseQuantizeConfig(
    bits=4,
    group_size=128,
    damp_percent=0.01,
    desc_act=False,
)
print(f"Loading model {model_name} ...\n")
model = AutoGPTQForCausalLM.from_pretrained(model_name, quantize_config, device_map="auto")
model.half()

tokenizer = AutoTokenizer.from_pretrained(model_name)

examples = [
    tokenizer(
        "Die Frage, was Wissenschaft ist und wie sie sich von anderen Bereichen menschlichen Handelns und menschlicher Errungenschaften unterscheidet, ist seit langem ein Gegenstand der Wissenschaftsphilosophie. Eine genaue und allgemein akzeptierte Definition findet sich in der Literatur nicht. Insbesondere das Abgrenzungsproblem, auch Demarkationsproblem genannt, welches die Abgrenzung von Wissenschaft gegenüber Pseudowissenschaft und Nichtwissenschaft beinhaltet, gilt nicht als abschließend geklärt. Einige Aspekte der Charakterisierung von Wissenschaft sind im Folgenden genannt.Die Wissenschaft ist ein System der Erkenntnisse über die wesentlichen Eigenschaften, kausalen Zusammenhänge und Gesetzmäßigkeiten der Natur, Technik, Gesellschaft und des Denkens, das in Form von Begriffen, Kategorien, Maßbestimmungen, Gesetzen, Theorien und Hypothesen fixiert wird."
    )
]

print(f"Quantizing {model_name} ...\n")
model.quantize(
    examples,
)
print(f"Saving quantized model {save_folder} ...\n")
model.save_quantized(save_folder, use_safetensors=True)
tokenizer.save_pretrained(save_folder)
