<div align="center">

<b>A Toolkit for Evaluating Large Vision-Language Models. </b>

English | [简体中文](./docs/zh-CN/README_zh-CN.md)

<a href="https://rank.opencompass.org.cn/leaderboard-multimodal">🏆 OC Learderboard </a> •
<a href="#%EF%B8%8F-quickstart">🏗️Quickstart </a> •
<a href="#-datasets-models-and-evaluation-results">📊Datasets & Models </a> •
<a href="#%EF%B8%8F-development-guide">🛠️Development </a>

<a href="https://huggingface.co/spaces/opencompass/open_vlm_leaderboard">🤗 HF Leaderboard</a> •
<a href="https://huggingface.co/datasets/VLMEval/OpenVLMRecords">🤗 Evaluation Records</a> •
<a href="https://huggingface.co/spaces/opencompass/openvlm_video_leaderboard">🤗 HF Video Leaderboard</a> •

<a href="https://discord.gg/evDT4GZmxN">🔊 Discord</a> •
<a href="https://www.arxiv.org/abs/2407.11691">📝 Report</a> •
<a href="#-the-goal-of-vlmevalkit">🎯Goal </a> •
<a href="#%EF%B8%8F-citation">🖊️Citation </a>
</div>

**VLMEvalKit** (the python package name is **vlmeval**) is an **open-source evaluation toolkit** of **large vision-language models (LVLMs)**. It enables **one-command evaluation** of LVLMs on various benchmarks, without the heavy workload of data preparation under multiple repositories. In VLMEvalKit, we adopt **generation-based evaluation** for all LVLMs, and provide the evaluation results obtained with both **exact matching** and **LLM-based answer extraction**.

## 🏗️ QuickStart

See [[QuickStart](./docs/en/Quickstart.md) | [快速开始](./docs/zh-CN/Quickstart.md)] for a quick start guide.

## 📊 Demonstration

```python
# Demo
from vlmeval.config import supported_VLM
model = supported_VLM['NEO1_0-2B-SFT']()
# Forward Single Image
ret = model.generate(['assets/apple.jpg', 'What is in this image?'])
print(ret)  # The image features a red apple with a leaf on it.
# Forward Multiple Images
ret = model.generate(['assets/apple.jpg', 'assets/apple.jpg', 'How many apples are there in the provided images? '])
print(ret)  # There are two apples in the provided images.
```