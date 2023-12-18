---
title:  "LoRA: LOW-RANK ADAPTATION OF LARGE LANGUAGE MODELS review"
toc: true
toc_sticky: true
permalink: /project/nlp/review/LoRA/
categories:
  - NLP
  - Paper Review
tags:
  - Large Language Model
  - PEFT
use_math: true
last_modified_at: 2023-07-05
---

```py
class LoRALayer():
    def __init__(
        self, 
        r: int, 
        lora_alpha: int, 
        lora_dropout: float,
        merge_weights: bool,
    ):
        self.r = r
        self.lora_alpha = lora_alpha
        # Optional dropout
        if lora_dropout > 0.:
            self.lora_dropout = nn.Dropout(p=lora_dropout)
        else:
            self.lora_dropout = lambda x: x
        # Mark the weight as unmerged
        self.merged = False
        self.merge_weights = merge_weights
```

## Introduction

LoRA는 

- ICLR: https://iclr.cc/virtual/2022/poster/6319
- OpenReview: https://openreview.net/forum?id=nZeVKeeFYf9

### Summary

전체 내용을 요약한다

- 문제는 뭐였고
- 이렇게 해서
- 저렇게 해결한다
- 결론은 어떻게 나온다.

## Challenges

많은 NLP 모델들이 _하나의_ 큰 PLM을 여러개의 downstream task에 fine-tuning하는 구조를 갖고 있다.
그러나 fine-tuning model의 주요 단점은 새로 학습 시킨 모델이 PLM만큼이나 많은 파라미터를 갖는다는 것이다.
GPT-2나 RoBERTa 수준에선 그저 "불편한" 문제였던 반면, 175B의 파라미터를 갖는 GPT-3에선 꽤나 중요한 문제가 되었다.
Few-shot learning이나 prompt engineering을 통해 좋은 성능을 얻을 수는 있지만, GPT-3 또한 fine-tuning할 경우 성능이 향상됨이 확인되었다.

![Fig.1-Fine-tuning-significantly-outperforms-few-shot-learning-on-GPT-3]({{site.url}}{{site.baseurl}}/assets/posts/nlp/LoRA-Fig.1.png){: .align-center}{: width="600"}

위는 GPT-3를 MNLI-matched 데이터에 학습시킨 결과로 fine-tuning을 통해 상당한 성능향상을 보이는걸 확인할 수 있다.

이러한 문제를 해결하기 위해 소량의 task-specific parameter만 학습하는 기법이 개발되었다.
그러나 모델의 깊이를 변경하거나 sequence length를 조절하는 과정에서 inference latency 등의 문제가 발생하였고, 무엇보다 중요한 것은 이러한 방법조차 fine-tuning baseline을 달성하는데 실패하거나 모델의 품질과 latency 사이의 트레이드 오프가 생기는 문제가 발생하였다.

## Related Work

LoRA의 연구 배경은 크게 Transformer Language Models, Prompt Engineering and Fine-Tuning, Parameter-Efficient Adaptation, Low-Rank Structures in Deep Learning으로 나뉜다.
이 중 language model 부분은 일반적인 NLP 논문들과 크게 다를 것이 없으므로 나머지를 집중적으로 확인해보자.

### Prompt Engineering and Fine-Tuning

GPT-3와 같은 LLM이 비록 약간의 training example만으로 few shot learning을 수행할 수 있더라도, 이는 prompt


- 뭐가 다르길래 latency가 없는지?

"The key functional difference is that our learned weights can be merged with the main weights during inference, thus not introducing any latency, which is not the case for the adapter layers (Section 3)"

## Contributions

이전 연구에서 over-parametrized model이 실은 low intrinsic dimension을 갖는다는 성질에 착안하여, model apatation에서 변화하는 weight 또한 낮은 "intrinsic rank"를 갖는다고 가정한다.

## Method

간략하게 전체 프로세스를 소개한다

- Row rank가 진짜 row rank인가?
- why decomposition?
- Why Gaussian?
- alpha/r
- full rank가 무슨 상관?
  - weight가 왜 full rank?
  - does not require the accumulated gradient update to weight matrices to have full-rank during adaptation.
- 트랜스포머 밸류랑 쿼리만 하는 이유 --> q, k vs v인 이유? q랑 k랑 비슷한 성질이 있나?
  - Adapting both Wq and Wv gives the best performance overall.
    - This suggests that even a rank of four captures enough information in ∆W such that it is preferable to adapt more weight matrices than adapting a single type of weights with a larger rank.
  - Table 6 shows that, surprisingly, LoRA already performs competitively with a very small r (more so for {Wq , Wv } than just Wq).
    - This suggests the update matrix ∆W could have a very small “intrinsic rank”
    - To further support this finding, we check the overlap of the subspaces learned by
different choices of r and by different random seeds. We argue that increasing r does not cover a
more meaningful subspace, which suggests that a low-rank adaptation matrix is sufficient.
- 단점이 잘 이해가 안감

### Detail-Method

구체적인 프로세스를 적는다.
제목은 논문에서 따온다.

We perform a sequence of empirical studies to answer the following questions: 
- 1. Given a parameter budget constraint, which subset of weight matrices in a pre-trained Transformer should we adapt to maximize downstream performance?
- 2. Is the “optimal” adaptation matrix ∆W really _rank-deficient_? If so, what is a good rank to use in practice?
- 3. What is the connection between ∆W and W ? Does ∆W highly correlate with W ? How large is ∆W comparing to W ?

> A matrix is said to be rank-deficient if it does not have full rank. The rank deficiency of a matrix is the difference between the lesser of the number of rows and columns, and the rank. 

## Experiment

실험과정에선 RoBERTa, DeBERTa, GPT-2에 대해 먼저 LoRA를 테스트하고, GPT-3 175B까지 스케일링하여 진행한다.

**FT<sup>Top2</sup>** : dapts just the last two layers on GPT2

We introduce **BitFit**, a sparse-finetuning method where only the bias-terms of the model (or a subset of them) are being modified. We show that with small-to-medium training data, applying BitFit on pre-trained BERT models is competitive with (and sometimes better than) fine-tuning the entire model. For larger data, the method is competitive with other sparse fine-tuning methods. Besides their practical utility, these findings are relevant for the question of understanding the commonly-used process of finetuning: they support the hypothesis that finetuning is mainly about exposing knowledge induced by language-modeling training, rather than learning new task-specific linguistic knowledge.

### Detail-Experiment

자세한 프로세스를 적는다.
제목은 논문에서 따온다.

We also replicate Houlsby et al. (2019) and Pfeiffer et al. (2021) according to their setup.

Q:
- we only report the typical standard deviation for a given task over random seeds, as opposed to providing one for every entry?? 그럼 원래는?
- 

## Conclusion

논문의 결론과 개인적인 소감, 아쉬운 점, 응용법 등을 정리한다.

![Fig.1-add-caption-here]({{site.url}}{{site.baseurl}}/assets/posts/CATEGORY/POST-NAME-Fig.1.png){: .align-center}{: width="600"}

![Caption](URL){: .align-center}{: width="600"}

