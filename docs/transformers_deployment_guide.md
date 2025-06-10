# 🚀 MiniMax Model Transformers Deployment Guide

[中文 Transformers 部署指南](./transformers_deployment_guide_cn.md)

## 📖 Introduction

This guide will help you deploy the MiniMax-Text-01 model using the [Transformers](https://huggingface.co/docs/transformers/index) library. Transformers is a widely used deep learning library that provides a rich collection of pre-trained models and flexible model operation interfaces with the following features:

- 🔥 Rich Ecosystem: It offers thousands of pretrained models for diverse NLP tasks, accessible via Hugging Face's Model Hub.
- ⚡ Unified API: Simplifies fine-tuning and inference with a consistent interface across architectures, reducing development complexity.
- 📦 Multilingual & Cross-Modal: Supports global languages and multimodal tasks (e.g., image-text), enabling versatile applications.
- ⚙️ Optimized Performance: Integrates with frameworks like PyTorch/TensorFlow and tools like vLLM for efficient deployment and scaling.

## Special Thanks

Special thanks to [Shakib](https://www.linkedin.com/in/shakibkhan66/) and [Armaghan](https://www.linkedin.com/in/armaghan-shakir/overlay/about-this-profile/) for helping MiniMax-Text-01 develop Transformers model support.

For more information on how to use MiniMax-Text-01 Transformers, you can also refer to the Transformers [model documentation](https://huggingface.co/docs/transformers/main/en/model_doc/minimax).

## 🛠️ Environment Setup

### Installing Transformers

```bash
pip install transformers torch accelerate
```

## 📋 Basic Usage Example

Because we haven't uploaded the code for MiniMaxAI/MiniMax-Text-01-hf yet, we need to modify the config.

The pre-trained model can be used as follows:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

MODEL_PATH = "{MODEL_PATH}"
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, device_map="auto", trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)

messages = [
 {"role": "user", "content": "What is your favourite condiment?"},
 {"role": "assistant", "content": "Well, I'm quite partial to a good squeeze of fresh lemon juice. It adds just the right amount of zesty flavour to whatever I'm cooking up in the kitchen!"},
 {"role": "user", "content": "Do you have mayonnaise recipes?"}
]

text = tokenizer.apply_chat_template(
 messages,
 tokenize=False,
 add_generation_prompt=True
)

model_inputs = tokenizer(text, return_tensors="pt").to(model.device)

generation_config = GenerationConfig(
 max_new_tokens=20,
 eos_token_id=tokenizer.eos_token_id,
 use_cache=True,
)

generated_ids = model.generate(**model_inputs, generation_config=generation_config)

generated_ids = [
 output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
]

response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
print(response)
```

## ⚡ Performance Optimization

### Speeding up with Flash Attention

The code snippet above showcases inference without any optimization tricks. However, one can drastically speed up the model by leveraging [Flash Attention](../perf_train_gpu_one#flash-attention-2), which is a faster implementation of the attention mechanism used inside the model.

First, make sure to install the latest version of Flash Attention 2 to include the sliding window attention feature:

```bash
pip install -U flash-attn --no-build-isolation
```

Also make sure that you have hardware that is compatible with Flash-Attention 2. Read more about it in the official documentation of the [Flash Attention repository](https://github.com/Dao-AILab/flash-attention). Additionally, ensure you load your model in half-precision (e.g. `torch.float16`).

To load and run a model using Flash Attention-2, refer to the snippet below:

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_PATH = "{MODEL_PATH}"
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, trust_remote_code=True, torch_dtype=torch.float16, attn_implementation="flash_attention_2", device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)

prompt = "My favourite condiment is"

model_inputs = tokenizer([prompt], return_tensors="pt").to("cuda")
generated_ids = model.generate(**model_inputs, max_new_tokens=100, do_sample=True)
response = tokenizer.batch_decode(generated_ids)[0]
print(response)
```

## 📮 Getting Support

If you encounter any issues while deploying the MiniMax-Text-01 model:
- Please check our official documentation
- Contact our technical support team through official channels
- Submit an Issue on our GitHub repository

We continuously optimize the deployment experience on Transformers and welcome your feedback!
