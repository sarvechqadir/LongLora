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

2. **Background**:
   - Large language models (LLMs) like LLaMA and LLaMA2 have predefined context sizes which limit their application in tasks like summarizing long documents or answering long questions.
   - The paper discusses the computational challenges of training LLMs with longer contexts and the inefficiencies of existing methods.
   - The paper introduces the concept of shift short attention (S2-Attn) as an efficient substitute for standard self-attention.

3. **LongLoRA Design**:
   - LongLoRA introduces shift short attention during fine-tuning but retains the original standard self-attention during inference.
   - The paper emphasizes the importance of trainable embedding and normalization layers for long context learning, even though they constitute a small proportion of the model's parameters.
   - 

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


## Critical Analysis:

**1. What was overlooked by the authors?**
One potential oversight could be the lack of comparison with other state-of-the-art methods that address similar challenges.

**2. What could have been developed further?**
The real-world applications and use-cases of LongLoRA could be explored in more depth. Demonstrating its efficacy in practical scenarios would provide more validation to the approach.
A more comprehensive analysis of the computational savings and performance trade-offs when using LongLoRA compared to other methods might add value.

**3. Were there any errors?**
No explicit errors were mentioned.

**4. Have others disputed the findings?**
There's no mention of disputes or criticisms.







