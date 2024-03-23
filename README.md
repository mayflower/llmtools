# llmtools
Some small and simple tools to work with llms

## add_chatml_tokens.py
This small script adds tokens needed for chatml to a huggingface model. 

It adds the tokens <|im_start|> and <|im_end|>, sets the eos_token to <|im_end|> and creates the according chatml template default in the tokenizer.

```bash
python add_chatml_tokens.py --model MY_MODEL_NaME --output_dir FOLDER_TO_STORE_MODIFIED_MODEL
```
### create_bf16.py
This small script converts existing models using fp32 or fp16 as the dtype to bf16.

```bash
python create_bf16.py --model MY_MODEL_NaME
```
### create_gptq.py.py
This small script quantizes existing models to GPTQ using AutoGPTQ.

```bash
python create_gptq.py --model MY_MODEL_NaME
```

### create_awq.py
This small script quantizes existing models to AWQ using AutoAWQ.

```bash
python create_awq.py --model MY_MODEL_NaME
```

### show_mtbench_results.py
This small script shows the evaluation results for the mt-bench-de benchmark to do a quick check for a new model.

```bash
python show_mtbench_results.py --model MY_MODEL_NaME
```




