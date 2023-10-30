# LongLoRA
This repository contains overview, explanation, and examples of LongLoRA as outlined in the official paper: [https://arxiv.org/pdf/2309.12307.pdf](https://arxiv.org/pdf/2309.12307.pdf)

** MLA Style**: 
Chen, Yukang, et al. "LongLoRA: Efficient Fine-tuning of Long-Context Large Language Models." arXiv preprint arXiv:2309.12307 (2023).

**APA Style**: 
Chen, Y., Qian, S., Tang, H., Lai, X., Liu, Z., Han, S., & Jia, J. (2023). LongLoRA: Efficient Fine-tuning of Long-Context Large Language Models. arXiv preprint arXiv:2309.12307.

## Overview

1. **Abstract and Introduction**:
   - The paper introduces LongLoRA, an efficient method for fine-tuning large language models (LLMs) to extend their context sizes without significant computational costs.
   - Training LLMs with long context sizes is usually computationally expensive. LongLoRA addresses this by using sparse local attention during fine-tuning and dense global attention during inference.
   - LongLoRA has been tested on various tasks and models, demonstrating strong results.
   - The paper also introduces a dataset, LongQA, for supervised fine-tuning containing over 3k long context question-answer pairs.
   - The code, models, dataset, and demo are available on GitHub.

![image](https://github.com/sarvechqadir/LongLora/assets/78235308/13fda6fb-1172-410f-870b-c1ae292ceb76)

2. **Background**:
   - Large language models (LLMs) like LLaMA and LLaMA2 have predefined context sizes which limit their application in tasks like summarizing long documents or answering long questions.
   - The paper discusses the computational challenges of training LLMs with longer contexts and the inefficiencies of existing methods.
   - The paper introduces the concept of shift short attention (S2-Attn) as an efficient substitute for standard self-attention.

3. **LongLoRA Design**:
   - LongLoRA introduces shift short attention during fine-tuning but retains the original standard self-attention during inference.
   - The paper emphasizes the importance of trainable embedding and normalization layers for long context learning, even though they constitute a small proportion of the model's parameters.

![image](https://github.com/sarvechqadir/LongLora/assets/78235308/77ecde7c-c491-4434-b63f-b9852af73c33)

     

4. **Related Work**:
   - The paper discusses various methods developed to increase the context length of transformers.
   - It also touches upon other efficient fine-tuning methods and their relevance to the current research.

5. **Shift Short Attention**:
   - The paper delves into the details of the proposed shift short attention mechanism, explaining its design and advantages over standard self-attention.
   - Shift short attention uses sparse local attention during fine-tuning instead of the usual dense global attention.
   - The input document is divided into distinct groups, with attention applied within each group.
   - For a model processing 8192 tokens, during self-attention, each group is restricted to 2048 tokens, resulting in 4 groups.
   - This method can face challenges with very long contexts, leading to increased perplexity due to limited information exchange between groups.
   - S2-Attn addresses this by shifting tokens by half the group size, ensuring better information flow between adjacent groups.
   - The output is combined coordinate-wise, using pre-trained self-attention weights.
   - The first and last 1024 tokens are in the same group, likely aiding in information exchange between the text's start and end.
   - Shift Short Attention efficiently extends context without added computational costs. Unlike standard self-attention with O(n²) computational cost, S2-Attn allows tokens to focus on nearby       tokens within a shifted window, making it efficient for long sequences.

   ![image](https://github.com/sarvechqadir/LongLora/assets/78235308/a13b579d-6f18-4e93-943e-e224269d93f2)



6. **Parameter-Efficient Fine-Tuning**:
   - LongLoRA's efficiency is boosted by rethinking the fine-tuning approach for context expansion.
   - LoRA, typically applied over attention layers, is effective when paired with embedding and normalization layers during training.
   - These components are vital for long-context learning but represent only a small portion of the model's parameters.
   - The inclusion of trainable embedding and normalization layers is key to LongLoRA's success.
   

## First Chosen Topic - Background:
Large language models like LLaMA and LLaMA2 have predefined context sizes which limit their application in tasks like summarizing long documents or answering long questions.
**Question for Class Discussion**: Why is it important to extend the context size of large language models, and what potential applications can benefit from it?

## Second Chosen Topic - LongLoRA Design:
LongLoRA introduces a new mechanism called shift short attention during fine-tuning but retains the original standard self-attention during inference.
**Question for Class Discussion**: How does the shift short attention mechanism in LongLoRA differ from standard self-attention, and what advantages does it offer?


##  Pseudocode Description of the Proposed Model:
This pseudocode provides a high-level overview of the LongLoRA fine-tuning process based on the summarized content. 

```
Algorithm: LongLoRA Fine-Tuning

Input: 
- Large Language Model (LLM) with predefined context size
- Training data with long context question-answer pairs

Output:
- Fine-tuned LLM with extended context size

1. Initialize LongLoRA with the base LLM parameters.
2. For each training data instance:
    a. Extract the long context and corresponding answer.
    b. Apply sparse local attention to the context during fine-tuning.
3. Retain the original standard self-attention mechanism for inference.
4. Introduce trainable embedding and normalization layers for long context learning.
5. Fine-tune the model using the LongQA dataset or other relevant long-context datasets.
6. Once fine-tuning is complete, use dense global attention for inference tasks.

End

```

---


---

## Results:

1. **Model Extensions**:
   - The authors expanded the pre-trained LLaMA2 models (7B, 13B, and 70B versions) to allow for larger context window sizes.

2. **Performance in Long-Sequence Language Modeling**:
   - Using the PG-19 and RedPajama datasets, longer sequences enhanced perplexity scores, showcasing LongLoRA's effectiveness. For example, the LLaMA2 7B model's perplexity reduced from 2.72 to 2.50 when the context window size increased from 8192 to 32768.

3. **Topic Retrieval Performance**:
   - In the LongChat dataset, LongLoRA matched the performance of the state-of-the-art model, LongChat-13B, across various conversation lengths. LongLoRA's efficiency was highlighted in its method of next-token generation using the RedPajama dataset, outperforming LongChat-13B in the 16k evaluation.

4. **Computation Analysis**:
   - As context length grows, the FLOPs ratio of attention in models without S2-Attn also increases. Introducing S2-Attn significantly reduces FLOPs, especially for longer context lengths. For instance, at a context length of 8192, attention accounts for 24.5% of total FLOPs without S2-Attn, but this ratio jumps to 72.2% at a context length of 65536. With S2-Attn, it drops to 39.4%.

---



## Critical Analysis:

**LongLora Advantages**:

1. **Preservation of Original Architecture**:
   - Models fine-tuned with S2-Attn maintain the original attention architecture during inference, facilitating the use of existing optimization techniques and infrastructure.

2. **Compatibility with Existing Techniques and Tools**:
   - LongLoRA works with attention optimization techniques like FlashAttention-2 during both training and inference, allowing researchers to integrate LongLoRA into their current workflows seamlessly.

3. **Easy Implementation**:
   - Implementing LongLoRA is simple, necessitating only two lines of code during training. There's also an option to keep the original standard self-attention of the trained model during inference.

4. **Train VS Inference Adaptation**:
   - LongLoRA provides an efficient solution for both training and inference, with the flexibility to maintain the original standard self-attention in the trained model during inference.



**1. What was overlooked by the authors?**
One potential oversight could be the lack of comparison with other state-of-the-art methods that address similar challenges.

**2. What could have been developed further?**
The real-world applications and use-cases of LongLoRA could be explored in more depth. Demonstrating its efficacy in practical scenarios would provide more validation to the approach.
A more comprehensive analysis of the computational savings and performance trade-offs when using LongLoRA compared to other methods might add value.

**3. Were there any errors?**
No explicit errors were mentioned.

**4. Have others disputed the findings?**
There's no mention of disputes or criticisms.

## Code Demonstration
Please refer to the [LongLoRA-notebook.ipynb](https://github.com/sarvechqadir/LongLora/blob/main/LongLoRA_notebook.ipynb) notebook.

## Resources
**Medium:**
   - [LongLoRA Explained: Efficient Fine-Tuning of Long Context LLMs](https://ai.plainenglish.io/longlora-how-to-extend-llms-context-sizes-through-fine-tuning-9f27894d1c06)

**Youtube:**
   - [How to code long-context LLM: LongLoRA explained on LLama 2 100K](https://youtu.be/hf5N-SlqRmA?si=gsqygN3LcmfE1qsp)
   - [LongLoRA: Efficient Fine-tuning of Long-Context Large Language Models](https://youtu.be/cftIv4DKu1E?si=SlwNQknD3SW_E8S6)

**Github**
   - [LongLoRA - Main GitHub repo](https://github.com/dvlab-research/LongLoRA/tree/main)

**HuggingFace**
   - [HuggingFace models](https://huggingface.co/Yukang)










