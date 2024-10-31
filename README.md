<h2 align="center"> <a href="https://arxiv.org/abs/2410.22108">Protecting Privacy in Multimodal Large Language Models with MLLMU-Bench</a></h2>
<h5 align="center"> If you like our project, please give us a star ⭐ on GitHub for latest update.  </h2>

<div align="center">    
<img src="./asset/demo.jpg" width="100%" height="50%">
</div>

## Abstract 
Generative models such as Large Language Models (LLM) and Multimodal Large Language models (MLLMs) trained on massive web corpora can memorize and disclose individuals' confidential and private data, raising legal and ethical concerns. While many previous works have addressed this issue in LLM via machine unlearning, it remains largely unexplored for MLLMs. To tackle this challenge, we introduce Multimodal Large Language Model Unlearning Benchmark (MLLMU-Bench), a novel benchmark aimed at advancing the understanding of multimodal machine unlearning. MLLMU-Bench consists of 500 fictitious profiles and 153 profiles for public celebrities, each profile feature over 14 customized question-answer pairs, evaluated from both multimodal (image+text) and unimodal (text) perspectives. The benchmark is divided into four sets to assess unlearning algorithms in terms of efficacy, generalizability, and model utility. Finally, we provide baseline results using existing generative model unlearning algorithms. Surprisingly, our experiments show that unimodal unlearning algorithms excel in generation and cloze tasks, while multimodal unlearning approaches perform better in classification tasks with multimodal inputs. 

## Quick Access:
- [Huggingface Dataset](https://huggingface.co/datasets/MLLMMU/MLLMU-Bench): Our benchmark is available on Huggingface. More updates comming soon. 
- [Arxiv Paper](https://arxiv.org/abs/2410.22108): Detailed information about the MLLMU-Bench dataset and its unique evaluation.
- [GitHub Repository](https://github.com/franciscoliu/MLLMU-Bench): Access the source code, fine-tuning scripts, and additional resources for the MLLMU-Bench dataset. You may also use our training data to fine-tune your own "vanilla" model!

## Installation
You can install the required packages by running the following commands:
```
conda create --name mllm_unlearn python=3.10
conda activate mllm_unlearn
pip install -r requirements.txt
```


## More details will be provided gradually throughout this week (Including detailed fine-tuning scripts, evaluation). Stay tuned, and sorry for the inconvenience!
