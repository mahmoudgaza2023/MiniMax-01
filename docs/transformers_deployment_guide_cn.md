# 🚀 MiniMax 模型 Transformers 部署指南

## 📖 简介

本指南将帮助您使用 [Transformers](https://huggingface.co/docs/transformers/index) 库部署 MiniMax-Text-01 模型。Transformers 是一个广泛使用的深度学习库，提供了丰富的预训练模型集合和灵活的模型操作接口，具有以下特性：

- 🔥 丰富的生态系统：通过 Hugging Face 模型中心提供数千个预训练模型，支持各种 NLP 任务。
- ⚡ 统一的 API：通过跨架构的一致接口简化微调和推理，降低开发复杂性。
- 📦 多语言和跨模态：支持全球语言和多模态任务（如图像-文本），实现多样化应用。
- ⚙️ 优化性能：与 PyTorch/TensorFlow 等框架以及 vLLM 等工具集成，实现高效部署和扩展。

## 特别感谢 

特别感谢 [Shakib](https://www.linkedin.com/in/shakibkhan66/) 和 [Armaghan](https://www.linkedin.com/in/armaghan-shakir/overlay/about-this-profile/) 帮助 MiniMax 开发了 Transformers 的算子支持。

关于 Transformers 的如何使用，也可以参考 Transformers 的[模型文档](https://huggingface.co/docs/transformers/main/en/model_doc/minimax)。

## 🛠️ 环境准备

### 安装 Transformers

```bash
pip install transformers torch accelerate
```

## 📋 基本使用示例

因为目前我们还未上传 MiniMaxAI/MiniMax-Text-01-hf 的代码，我们需要修改config。

预训练模型可以按照以下方式使用：

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

## ⚡ 性能优化

### 使用 Flash Attention 加速

上面的代码片段展示了不使用任何优化技巧的推理过程。但通过利用 [Flash Attention](../perf_train_gpu_one#flash-attention-2)，可以大幅加速模型，因为它提供了模型内部使用的注意力机制的更快实现。

首先，确保安装最新版本的 Flash Attention 2 以包含滑动窗口注意力功能：

```bash
pip install -U flash-attn --no-build-isolation
```

还要确保您拥有与 Flash-Attention 2 兼容的硬件。在[Flash Attention 官方仓库](https://github.com/Dao-AILab/flash-attention)的官方文档中了解更多信息。此外，请确保以半精度（例如 `torch.float16`）加载模型。

要使用 Flash Attention-2 加载和运行模型，请参考以下代码片段：

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

## 📮 获取支持

如果您在部署 MiniMax-Text-01 模型过程中遇到任何问题：
- 请查看我们的官方文档
- 通过官方渠道联系我们的技术支持团队
- 在我们的 GitHub 仓库提交 Issue

我们会持续优化 Transformers 上的部署体验，欢迎您的反馈！