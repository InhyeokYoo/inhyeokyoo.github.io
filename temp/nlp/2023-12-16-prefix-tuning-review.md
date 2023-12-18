---
title:  "Prefix-Tuning: Optimizing Continuous Prompts for Generation review"
toc: true
toc_sticky: true
permalink: /project/nlp/review/prefix-tuning
categories:
  - NLP
  - Paper Review
tags:
  - Large Language Model
  - PEFT
use_math: true
last_modified_at: 2023-12-16
---

## Introduction

간략한 개요와 참고 리소스 등을 적는다.

https://aclanthology.org/2021.acl-long.353.pdf


### Summary

전체 내용을 요약한다

- 문제는 뭐였고
- 이렇게 해서
- 저렇게 해결한다
- 결론은 어떻게 나온다.

## Challenges

Consequently, to build and deploy NLP systems
that rely on large pretrained LMs, one currently
needs to store a modified copy of all the LM pa-
rameters for each task. This can be prohibitively
expensive given the size of current LMs; for exam-
ple, GPT-2 has 774M parameters (Radford et al.,
2019) and GPT-3 has 175B parameters (Brown
et al., 2020).





## Related Work

A natural approach to this problem is lightweight
fine-tuning, which freezes most of the pretrained
parameters and only tunes a smaller set of param-
eters.

For example, adapter-tuning (Rebuffi et al., 2017; Houlsby et al., 2019) inserts additional task-
specific layers between the layers of pretrained
language models. Adapter-tuning has promising
performance on natural language understanding
and generation benchmarks, attaining comparable
performance with fine-tuning while adding only
around 2–4% task-specific parameters (Houlsby
et al., 2019; Lin et al., 2020)

At the limit, GPT-3 (Brown et al., 2020) can
be deployed using in-context learning, which is
a form of prompting, without modifying any LM
parameters. In in-context learning, Brown et al.
(2020) prepend a natural language task instruction
(e.g., TL;DR for summarization) and a few exam-
ples to the task input, and then generate the task
output from the LM. However, since Transformers
can only condition on a bounded-length context
(e.g., 2048 tokens for GPT-3), in-context learning
is restricted to very small training sets.

### Fine-tuning for natural language generation

Current state-of-the-art systems for natural lan-
guage generation (NLG) are based on fine-tuning
pretrained LMs. For table-to-text generation, Kale
(2020) fine-tunes a sequence-to-sequence model
(T5; Raffel et al., 2020). For extractive and abstrac-
tive summarization, researchers fine-tune masked
language models (e.g., BERT; Devlin et al., 2019)
and encode-decoder models (e.g., BART; Lewis
et al., 2020), respectively (Zhong et al., 2020; Liu
and Lapata, 2019; Raffel et al., 2020). For other
conditional NLG tasks such as machine transla-
tion and dialogue generation, fine-tuning is also the
prevalent paradigm (Zhang et al., 2020c; Stickland
et al., 2020; Zhu et al., 2020; Liu et al., 2020). In
this paper, we focus on table-to-text using GPT-2
and summarization using BART, but prefix-tuning
in principle can be applied to other generation tasks
and pretrained models, such as masked LMs.

### Lightweight fine-tuning

Prefix-tuning falls under the broad class of lightweight fine-tuning methods, which freeze most of the pretrained parameters and only tune a smaller set of parameters.
The key question is how to augment the LM architecture and decide which subset of pretrained parameters to tune.
One line of research learns a task-specific parameter mask (Zhao et al., 2020; Radiya-Dixit and Wang, 2020).

Another line of research inserts new modules with trainable parameters.
For example, Zhang et al. (2020a) trains a **side** network that is fused with the pretrained model via summation; adapter-tuning inserts task-specific layers (adapters) between each layer of the pretrained LM (Houlsby et al., 2019; Lin et al., 2020; Rebuffi et al., 2017; Pfeiffer et al., 2020).
Compared to this line of work, which tunes around 3.6% of the LM parameters, our method obtains a further 30x reduction in task-specific parameters, tuning only 0.1% while maintaining comparable performance on table-to-text tasks.

### Prompting

Prompting is a way of leveraging a pretrained LM by prepending instructions and a few examples to the task input and generating the task output from the LM.
For autoregressive LMs, the most successful form of prompting is GPT-3’s in-context learning (Brown et al., 2020), which uses manually designed prompts to adapt its generation for different tasks in few-shot settings.
For masked LMs like BERT and RoBERTa (Liu et al., 2019), prompt engineering has been explored for natural language understanding tasks (Jiang et al., 2020; Schick and Sch ¨utze, 2020).
For example, AutoPrompt (Shin et al., 2020) searches for a sequence of discrete trigger words and concatenates it with each input to elicit sentiment or factual knowledge from BERT and RoBERTa.
In contrast with AutoPrompt, our method optimizes continuous prefixes, which are more expressive (§7.2); moreover, we focus on language generation tasks.

Continuous vectors have been used to steer LMs; for example, Subramani et al. (2020) showed that a pretrained LSTM language model can reconstruct arbitrary sentences by optimizing a continuous vec tor for each sentence, making the vector input-specific.
In contrast, prefix-tuning optimizes a task-specific prefix that applies to all instances of that task.
As a result, unlike the previous work whose application is limited to sentence reconstruction, prefix-tuning can be applied to NLG tasks.

### Controllable generation

Controllable generation aims to steer a pretrained language model to match a sentence-level attribute (e.g., positive sentiment or sports).
Such control can happen at training time: Keskar et al. (2019) pretrains the language model (CTRL) to condition on metadata such as keywords or URLs.
The control can also happen at decoding time, by weighted decoding (GeDi, Krause et al., 2020) or iteratively updating the past activations (PPLM, Dathathri et al., 2020).
However, there is no straightforward way to apply these controllable generation techniques to enforce fine-grained control over generated contents, as demanded by tasks like table-to-text and summarization.

### P*-tuning

Prefix tuning is an instance of a new class of methods that has emerged, which we call p*-tuning (since the other prominent instances, p-tuning and prompt-tuning, also start with p), all based on the idea of optimizing a continuous prefix or prompt.
Concurrent with our work, Qin and Eisner (2021) (soft prompt) learn mixtures of soft fill-in-the-blank prompts to elicit knowledge from LMs such as BERT and BART.
Hambardzumyan et al. (2021) (WARP) learns task-specific embeddings that adapts BERT for sentiment classification.
Both works show that tuning soft prompts outperforms previous work, which optimizes over discrete prompts.
P-tuning (Liu et al., 2021) shows that jointly updating the prompt embeddings and LM parameters improves GPT-2’s performance on natural language understanding tasks, in both few-shot and full data settings.
In a followup work, Prompt-tuning (Lester et al., 2021) simplifies our approach and applies it to T5 (Raffel et al., 2020), demonstrating that the performance gap between fine-tuning and p*-tuning vanishes as the model size grows.

## Contributions

In this paper, we propose prefix-tuning, a
lightweight alternative to fine-tuning for natural lan-
guage generation (NLG) tasks, inspired by prompt-
ing.
Consider the task of generating a textual de-
scription of a data table, as shown in Figure 1,
where the task input is a linearized table (e.g.,
“name: Starbucks | type: coffee shop”) and the out-
put is a textual description (e.g., “Starbucks serves
coffee.”).

![Figure 1]({{site.url}}{{site.baseurl}}/assets/posts/nlp/prefix-tuning-Fig.1.png){: .align-center}{: width="600"}

Prefix-tuning prepends a sequence of
**continuous task-specific** vectors to the input, which
we call a prefix, depicted by red blocks in Figure 1
(bottom).
To generate each token, the LM can attend to the prefix as if it were a sequence of "virtual
tokens", but unlike prompting, the prefix consists
entirely of free parameters which do not correspond
to real tokens.
In contrast to fine-tuning in Figure 1
(top), which updates all LM parameters and thus
requires storing a tuned copy of the model for each
task, prefix-tuning only optimizes the prefix.
Consequently, we only need to store one copy of the
large LM and a learned task-specific prefix, yield-
ing a very small overhead for each additional task
(e.g., 250K parameters for table-to-text).

In contrast to full fine-tuning, prefix-tuning is
also modular: we train an upstream prefix which
steers an unmodified LM, and therefore, a single
LM can support many tasks at once. In the con-
text of personalization where the tasks correspond
to users (Shokri and Shmatikov, 2015; McMahan
et al., 2016), we would have a separate prefix for
each user trained only on that user’s data, thereby
avoiding data cross-contamination. Moreover, the
prefix-based architecture enables us to even pro-
cess examples from multiple users/tasks in a single
batch, something that is not possible with other
lightweight fine-tuning approaches like adapter-
tuning.

We evaluate prefix-tuning on table-to-text gen-
eration using GPT-2 and abstractive summariza-
tion using BART. In terms of storage, prefix-tuning
stores 1000x fewer parameters than full fine-tuning.
In terms of performance when trained on full
datasets, prefix-tuning and fine-tuning are compara-
ble for table-to-text (§6.1), while prefix-tuning suf-
fers a small degradation for summarization (§6.2).
In low-data settings, prefix-tuning outperforms fine-
tuning on both tasks (§6.3). Prefix-tuning also ex-
trapolates better to tables (for table-to-text) and arti-
cles (for summarization) with unseen topics (§6.4).

## Method

![Figure 2]({{site.url}}{{site.baseurl}}/assets/posts/nlp/prefix-tuning-Fig.2.png){: .align-center}{: width="800"}

### Dataset

T5와 마찬가지로 SuperGlue를 text-to-text 형태로 변형하여 사용하되, input 앞에 붙는 prompt에 SuperGLUE task는 제외하여 어떠한 형태의 데이터인지는 알 수 없게 만들었다.

### Hyper Paramters

prompt: 30,000 step + CE Loss + learning rate 0.3 + batch size 32 + Adafactor optimizer (weight decay $1e-5$, $\beta _2$ decay 0.8)

Early stopping 적용

TODO: 자세한 내용은 appendix A에있음

## Experiment

### Ablation Study: Prompt Length


## Conclusion

논문의 결론과 개인적인 소감, 아쉬운 점, 응용법 등을 정리한다.

![Fig.1-add-caption-here]({{site.url}}{{site.baseurl}}/assets/posts/CATEGORY/POST-NAME-Fig.1.png){: .align-center}{: width="600"}

![Caption](URL){: .align-center}{: width="600"}

