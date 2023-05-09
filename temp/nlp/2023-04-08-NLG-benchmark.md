---
title:  "텍스트 생성 시 디코딩 방법"
toc: true
toc_sticky: true
permalink: /project/nlp/decoding-strategy/
categories:
  - NLP
tags:
  - NLU
  - Generative Model
use_math: true
last_modified_at: 2023-04-08
---

## 들어가며


## Greedy search

## Beam search


Low k → greedy decoding과 유사해짐, More on-topic but nonsensical

High k → greedy decoding의 문제점 해결. Safe and correct response but less relevant(generic) 지나치게 큰 k 값은 오히려 BLEU 점수 감소를 야기.

Good generation ↔ Computing cost (Trade-off)


*Open-ended generation*에서 beam search는 잘 동작하지 않는데, 그 이유는 다음과 같다.

> The goal in *open-ended text generation* is to create a coherent portion of text that is a continuation from the given context. For example, given a couple of sentences, this capability makes it possible for machines to self-write a coherent story. One can imagine using such a system for AI-assisted writing, but of course it can also be repurposed to generate misleading (fake) news articles. 
>
> [https://blog.fastforwardlabs.com/2019/05/29/open-ended-text-generation.html](https://blog.fastforwardlabs.com/2019/05/29/open-ended-text-generation.html)

- 요약/NMT같이 바람직하게 생성된 텍스트의 길이가 다소 예측 가능한 경우에는 잘 동작하지만 대화/이야기 생성과 같은 open-ended generation의 경우 바람직한 아웃풋의 길이가 매우 상이하다.
- 텍스트를 생성할 때 특정 문구가 반복되는 것을 볼 수 있는데, 이를 컨트롤하기가 까다롭다.
- 고품질의 인간 언어와 next word에 대한 확률분포와는 다소 차이가 있다. 우리는 생성된 텍스트가 지루하거나 예측가능하기보단 놀랍고 예측불가능하기를 원한다.

It turns out likelihood maximization approaches such as beam search tend to produce sentences that loop repetitively. Further, the probability of forming a loop (“I don’t know, I don’t know, I don’t know”) increases with a longer loop - once looping starts, it is difficult to get out of it. In addition, probability distribution of human-generated text turns out to be very different from machine-generated text. When using a maximum likelihood framework, the machine-generated text is composed of tokens that are highly probable, but human-generated text exhibits much richness and variance.

이를 위해서 sampling을 통해 randomness를 추가할 수 있다.

## Top-K sampling

## Top-p sampling

## Sampling temperature

However, recent analysis has shown that, while lowering the
temperature improves generation quality, it comes at the cost of decreasing diversit

## Generative Model

생성 모델의 경우 `GenerationMixin` 클래스의 `generate()` 함수를 사용한다.
생성에 필요한 파라미터는 [`GenerationConfig`](https://huggingface.co/docs/transformers/v4.27.2/en/main_classes/text_generation#transformers.GenerationConfig) 클래스의 인스턴스를 통해 넣어 이를 제어할 수 있다.
아니면 `generate()`를 호출할 때 값을 넣어줘도 무방하다.

### `generate()`



Most generation-controlling parameters are set in generation_config which, if not passed, will be set to the model’s default generation configuration. You can override any generation_config by passing the corresponding parameters to generate(), e.g. .generate(inputs, num_beams=4, do_sample=True).

For an overview of generation strategies and code examples, check out the following guide.

`generate` 함수:
- `max_new_tokens`: 생성되는 토큰의 최대치
- `do_sample`: if set to True, this parameter enables decoding strategies such as multinomial sampling, beam-search multinomial sampling, Top-K sampling and Top-p sampling. All these strategies select the next token from the probability distribution over the entire vocabulary with various strategy-specific adjustments.
- `no_repeat_ngram_size`: If set to int > 0, all ngrams of that size can only occur once. TODO: 원리가 뭐지??
- `tempature`: softmax 적용 전 logit의 scale을 조정: $\frac{\exp(z _{t, i}/T)}{\sum^{|V|} _{j=1} \exp(z _{t, i}/T)}$. 1보다 작으면 원점 근처에서 최댓값이 되고, 희귀한 토큰이 나올 확률을 감소. 1보다 크면 분포가 평평해지며 각 토큰의 확률이 비슷해짐 (따라서 횡설수설에 가까운 텍스트가 생성). 이에 대한 대안으로는 Top-k/Top-p sampling이 있음
- `num_return_sequences`: beam search로 생성할 문장의 갯수. `num_return_sequences` <= `num_beams`이여야 한다.

수백 번 샘플링하게 되면 언젠가 희귀한 토큰을 선택할 수 있음.
희귀한 단어를 생성하게 되는 경우 생성된 텍스트의 품질이 떨어질 가능성이 있음.
따라서 희귀한 단어를 피하기 위해 Top-k/Top-p sampling 사용

Top-k sampling

확률이 가장 높은 K개 토큰에서만 샘플링하여 희귀한 토큰을 피함.
K값은 수동으로 지정하며 매 스텝에서 일관되게 적용.


Top-p sampling

누적확률값이 P가 되는 토큰들만 골라 샘플링.
Top-K와는 달리 동적임


- Greedy search decoding: 반복적인 출력 시퀀스를 생성하는 경향이 있어 다양성이 필요한 작업에는 사용되지 않으나, 결정적이고 사실적으로 정확한 출력이 필요한 수식 등의 짧은 문장 생성에는 유용
- Beam search decoding: 로그확률로 표현


{: .align-center}{: width='700'}


## 참고자료

1. Platen, P. (2020, March 1). *How to generate text: using different decoding methods for language generation with Transformers*. Hugging Face. [https://huggingface.co/blog/how-to-generate?fbclid=IwAR19kbEiW_sF19TeSr4BE4jQZSIqz0GzOFD2013fIGEH32DReW9pAFq6vDM](https://huggingface.co/blog/how-to-generate?fbclid=IwAR19kbEiW_sF19TeSr4BE4jQZSIqz0GzOFD2013fIGEH32DReW9pAFq6vDM)
2. https://arxiv.org/pdf/1904.09751.pdf
3. https://huggingface.co/blog/how-to-generate?fbclid=IwAR19kbEiW_sF19TeSr4BE4jQZSIqz0GzOFD2013fIGEH32DReW9pAFq6vDM
4. https://blog.fastforwardlabs.com/2019/05/29/open-ended-text-generation.html
5. https://huggingface.co/docs/transformers/v4.27.2/en/generation_strategies#contrastive-search

