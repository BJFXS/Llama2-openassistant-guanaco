# Llama2-openassistant-guanaco
Llama2-OpenAssistant-Guanaco is a fine-tuning project designed to enhance the text-generation abilities of a simplified Llama-2 model. Llama-2 is a powerful pretrained large language model, and this project adapts it using the OpenAssistant-Guanaco dataset — a diverse, multilingual, instruction-following corpus.

The goal of this fine-tuning work is to improve the model’s fluency, accuracy, and generalization across multiple domains. By training on varied dialog-style data, the resulting model becomes more aligned, more conversational, and better prepared for real-world question-answering tasks.

> **Note:** The original training notebook (`.ipynb`) cannot be previewed directly on GitHub. You may download it and open it in **Google Colab**. The included `llama_2_openassistant.py` file is an auto-converted version for easy browsing.


##  Features

**4-bit Quantized Loading (bitsandbytes)** 

Efficient model loading using **NF4 4-bit quantization**, enabling training on limited GPU memory.

**LoRA Fine-Tuning (PEFT)**

Lightweight parameter-efficient fine-tuning via **LoRA**, making it possible to fine-tune a 7B model on a single GPU.

**TRL SFT Trainer**

Uses HuggingFace's **Supervised Fine-Tuning (SFT) Trainer** to handle packing, batching, evaluation, checkpoints, and more.

**HuggingFace Hub Integration**

After training, the LoRA weights are **merged** into the base model and uploaded automatically to the user’s HuggingFace account.

**Post-Training Inference Tests**

The script downloads the fine-tuned model and evaluates it using various prompts (economics, ML definitions, history, creative writing, DevOps, etc.).





## Installation & Setup

**Install all required dependencies:**

```bash
pip install -q huggingface_hub
pip install -q -U trl transformers accelerate peft
pip install -q -U datasets bitsandbytes einops wandb
pip install -q ipywidgets scipy
```

**Log into HuggingFace:**

from huggingface_hub import notebook_login
notebook_login()


## Dataset

The project fine-tunes Llama-2 on:

```bash
timdettmers/openassistant-guanaco
```



## Training the Model

The training process:

```bash
trainer = SFTTrainer(
    model=base_model,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    peft_config=peft_config,
    packing=True,
    tokenizer=tokenizer,
    args=training_args,
)
trainer.train()
trainer.save_model(output_dir)
```



## Uploading the Model to HuggingFace

After merging LoRA with the base weights:

```bash
merged_model.push_to_hub("llama2-7b-finetunned-openassistant-origin")
```



 ## Testing the Fine-Tuned Model

Example generation after fine-tuning:

```bash
encoding = tokenizer(prompt, return_tensors="pt").to("cuda:0")
output = base_model.generate(max_new_tokens=400, temperature=0.000001)
print(tokenizer.decode(output[0]))
```



## Running on Google Colab (Recommended)

Because the project was originally created in a .ipynb notebook:

How to run it yourself:

1. Download the original notebook from GitHub.

2. Upload it to Google Colab.

3. Enable GPU (Runtime → Change runtime type → GPU).

4. Run all cells sequentially.

> **Note:** The .py file shown here contains the full code extracted from the notebook and is provided for convenient viewing inside GitHub.





## Acknowledgements

Meta AI for Llama-2

HuggingFace Transformers, TRL, PEFT (https://huggingface.co/blog/dpo-trl) (https://github.com/huggingface/trl/blob/main/examples/)

BitsAndBytes (4-bit quantization)

OpenAssistant project for dataset resources
