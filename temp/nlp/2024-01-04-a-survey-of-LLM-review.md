---
title:  "내 멋대로 하는 A Survey of Large Language Models review"
toc: true
toc_sticky: true
categories:
  - NLP
  - Paper Review
tags:
  - Instruction Tuning
  - Large Language Model
use_math: true
last_modified_at: 2024-01-04
---

## 들어가며

다음은 [A Survey of Large Language Models](https://arxiv.org/abs/2303.18223?fbclid=IwAR1o9DcsIuJ-_ZBHl8z7PWpxUDfTbGDHr_Drb2w3JtC5cfuE07na7q1Zhsw&mibextid=S66gvF)의 section 일부분 중 내가 필요한 내용만을 발췌하여 정리한 것이다.

또한 서베이답게 다양한 연구가 등장하기 때문에 모든 연구가 아닌 **내가 추후 살펴볼 가능성**이 있을만한 연구에만 reference를 달았다. Citation의 표기는 원문 그대로를 따랐지만, 본문과 비교하여 확인하는 것이 더욱 좋겠다.

아직 작성 중이므로, 정리 도중 멈춘 부분이나, 포스팅 용 템플릿이 그대로 남은 경우가 있을 것이다.

## 개요

LLM을 pre-train하면 다양한 task를 쉽게 풀 수 있는 일반화 성능을 얻게 되지만, 많은 연구에서 LLM을 특정 목적에 맞게 adaptation하는 경우 성능이 더욱 증가함을 보이고 있다.
이러한 성능 향상을 위한 두 가지 방법으로는 **instruction tuning**과 **alignment tuning**이 있다.

전자의 경우 LLM의 성능을 향상시키거나 잠금해제하는 것을 목표로하며, 후자의 경우 LLM의 동작을 사람의 preference와 value에 맞게 align하는 것이다.

## Instruction Tuning

본질적으로 instruction tuning은 PLM에 대한 fine-tuning이며, InstructGPT에서 나온 supervised fine-tuning (SFT)와 multi-task prompted training과 매우 연관성이 깊다.
Instruction tuning을 하면 unseen data에 대한 일반화 성능이 우수한 것으로 증명되었으며, 심지어는 **multilingual setting**에서도 좋은 것으로 드러났다 \[[94 (BLOOMZ-P3)][94]\].

해외에 비해 한국어 데이터셋이 부족한 실정인데 \[[94 (BLOOMZ-P3)][94]\]나 다른 논문을 통해 multilingual 세팅에서 성능이 향상되는지 확인하면 좋을듯 싶다.
{: .notice--warning}

또한, \[[342][342]\]와 같은 **instruction tuning에 관련된 체계적인 개요**를 나타낸 논문도 있는반면 본 논문에서는 instruction tuning에 대한 영향과 튜닝 및 데이터셋 생성에 대한 자세한 가이드라인, 전략을 다루고 있다.

\[[342][342]\]도 좋은 스타트 포인트가 될듯하다.
{: .notice--warning}


### Formatted Instance Construction

일반적으로 데이터는 다음과 같이 이루어진다.
- task description (instruction)
- optional input
- corresponding output
- (optional) a small number of *demonstrations*

*demonstrations*은 in-context learning에서 사용하는 input/output을 이야기 하는 것으로 보인다.
{: .notice--info}

#### Formatting NLP Task Datasets

이 방법은 labeled dataset에 인간이 작성한 task descriptions을 더해 LLM을 학습시키는 것이다.

![Fig.1-Formatting-Task-Datasets]({{site.url}}{{site.baseurl}}/assets/posts/nlp/a-survey-of-LLM-review-Fig.1){: .align-center}{: width="500"}

위 그림에서는 "Please answer this question"가 QA task의 task description이 된다.
Instruction tuning을 수행하면 task description을 따라 unseen data에 대한 성능이 높아지며, 없는 경우 성능이 심각하게 하락하기도 한다.

[PromptSource](https://github.com/bigscience-workshop/promptsource)와 같은 크라우드 소싱 플랫폼을 통해 데이터셋을 생성할 수도 있다.

데이터양을 늘리기 위해 기존 데이터셋의 input-output을 역으로 바꾸어 (e.g., "Please generate a question based on the answer:") 데이터를 생성하는 케이스도 있다 \[[28 (T0)][28], [168 (MVP)][168], [345 (The Flan Collection)][345] \]

\[[28 (T0)][28], [168 (MVP)][168], [345 (The Flan Collection)][345] \]같은 논문은 회사사람들과 공유하는 것도 좋아보인다.
{: .notice--warning}

#### Formatting Daily Chat Data

ChatGPT와 같이 사람들을 통해 직접 데이터를 생성한다.
Task의 다양성을 늘리기 위해 현실에서 사용할 법한 open-ended generation, QA, 브레인스토밍, 채팅과 같은 instruct를 제시하기도하며, 이를 토대로 다른 labeler가 적절한 답변을 생성하게한 후 학습한다.
(다만 InstructGPT는 이를 통해 alignment model도 학습시킨다)
더 나아가 GPT-4의 경우엔 부당한 instructions을 거부하게끔 학습시키기도 한다.

이러한 데이터셋은 공개되지 않는 경우가 많기 때문에 실제 유저의 질문에 대해 GPT-4/ChatGPT를 이용하여 input-output 데이터셋을 생성하는 경우 (e.g. ShareGPT)도 있으며, Dolly와 OpenAssistant와 같이 좋은 퀄리티임에도 불구하고 공개한 데이터셋도 존재한다.

#### Formatting Synthetic Data

Self-Instruct와 같이 인공적으로 데이터를 생성하는 방법이 이에 포함된다.
그러나 이러한 데이터는 단순하거나 다양성이 부족하며, WizardLM과 같이 Evol-Instruct 방법을 통해 이러한 측면을 채워주기도 한다.
또한, Self-Align의 경우 multiple human-aligned principle을 통해 합성 데이터를 필터링한 후 학습하기도하며, 사람이 작성한 output을 토대로 ICL을 통해 instruction을 생성하는 방법도 있다 \[[348][348]\].

#### Key Factors for Instance Construction

주요점을 정리하면 다음과 같다:
1. Scaling the instructions
  - 학습에 사용되는 task의 수가 많아야 좋다 \[[28 (T0)][28], [67][67], [88][88]\].
  - 질적, 구조적, 창의적 측면에서 instruct의 다양성을 높여야 한다 \[[69 (FLAN-T5)][69], [88][88]\].
  - 그러나 task 당 instance의 수는 낮은 것이 좋다 (1K-2K로도 좋은 성능을 보였음) \[[349 (LIMA)][349], [350 (ALPAGASUS)][350]\].
  - 반대로 데이터 양을 통해 성능을 향상시킨 경우도 있었으며 \[[351 (Orca)][351], [352][352]\], 특히 Orca의 경우 5M까지 step-by-step explanation으로 인공 데이터의 수를 늘려 좋은 성능을 보였다.
2. Formatting desing
  - natural language format (특히 task description)을 잘 만드는 것이 매우 큰 영향을 준다 \[[88][88]\].
  - 적절한 수의 demonstration를 함께 넣어주면 민감도를 낮추는 효과가 있다 \[[67][67], [69][69]\]. 
  - 단, 피해야할 것, 이유, 제안 등은 별 효과가 없거나 오히려 성능일 낮추었다 \[[88][88], [166][166]\].
  - 산술추론에서 CoT 사용여부와 성능엔 큰 영향이 없었다. 심지어 reasoning이 필요한 task에서도! \[[69]\]

결론내면 다양성이나 질적 측면이 데이터의 수보다 매우 중요하지만 \[[349][349]\], 이러한 데이터가 없다면 Orca처럼 압도적인 양으로 찍어누르는 것이 좋다.

### Instruction Tuning Strategies

#### Balancing the Data Distribution

다양한 task의 데이터셋이 있으므로 이를 잘 샘플링하는 것도 중요할 것이다.
널리 쓰이는 방법으로는 **examples-proportional mixing strategy**가 있다.
이는 단순히 데이터셋을 전부 섞은뒤 뽑는 것이다.
또한 좋은 퀄리티의 데이터셋의 ratio는 높게 뽑는 것도 성능 향상에 도움이 되며, **maximum cap**으로 제한을 걸어 큰 데이터셋이 성능을 좌지우지하는 것을 막는 것도 일반적으로 사용된다.
보통은 몇천에서 몇만사이로 세팅한다 \[[67][67], [69][69]\].

최근에는 아래 Table 3의 instruct dataset이 특정 측면에서의 LLM 성능을 향상시키는 데 중점을 두고 있으며, single dataset 하나만으로는 model capacity의 포괄적인 향상으로 이어질 수 없다는 것이 밝혀졌다 \[[353][353]\].
따라서 NLP task data (e.g., FLAN v2), chat data (e.g., ShareGPT), and synthetic data (e.g., GPT4-Alpaca)를 혼합하여  다양한 capacity에서 균형 잡힌 개선을 달성하도록 제안된다.

![Fig.2-Table-3]({{site.url}}{{site.baseurl}}/assets/posts/nlp/a-survey-of-LLM-review-Fig.2.png){: .align-center}{: width="400"}

#### Combining Instruction Tuning and Pre-Training

Tuning을 더욱 효율적이고 안정하게 만들기위해 OPT-IML같은 경우는 pre-training data를 섞기도 한다.
이는 일종의 regularization으로 사용한다.
또한, pre-training + instruction tuning의 두 단계를 거치는 것이 아닌 두 데이터셋을 pre-training 단계부터 multi-task learning 형태로 쓰는 경우도 존재한다.

#### Multi-stage Instruction Tuning

Instruction tuning에는 **task-formatted instruction**와 **daily chat instruction**의 두 종류의 중요한 데이터셋이 있다.
일반적으로는 전자가 후자보다 훨씬 크며, 두 종류의 데이터셋 사이의 균형을 맞추는 것이 중요하다.

그러나 이런 방법말고 large-scale task-formatted instruction에 LLM을 학습한 후 daily chat instruction을 학습하는 방법도 존재하는데, 이를 **Multi-stage Instruction Tuning**이라 한다 \[[352][352]\].
*Capacity forgetting* 을 피하기 위해 task-formatted instruction을 두번째 stage에 소량 추가하기도 한다.

*Capacity forgetting*은 아무래도 catastrophic forgetting인데 오타가 난듯 싶다.
{: .notice--info}

이러한 방법론은 다른 instruction tuning setting을 적용할 때 유용한데, 예를들어 학습이 진행될 수록 점차 어렵고 복잡한 데이터를 추가하여 학습을 원활하게 하는 것이 그 예일 것이다.

#### Other Practical Tricks

이 외에도 실전에서 사용할만한 여러 팁들이 있다.

1. Efficient training for multi-turn chat data
  - Multi-turn 데이터셋을 여러개의 context-response pairs로 나누어 학습하여 하나의 context와 답변만을 학습하게 만드는 것이다.
  - 이 경우 중복된 utterance를 만들 수 있다.
  - training cost를 줄이기 위해서는 Vicuna와 같이 모든 대화를 한 번에 집어넣되 챗봇의 response에 대해서만 loss를 계산하는 방법도 존재한다.
2. Establishing self-identification for LLM
  - Real-world에 deploy하려면 이의 소속이나 개발자, 기관 등을 인식시켜줘야 함.
  - 이를 위해 identity-related instruction가 필요
  - 혹은 self-identification prompt ("The following is a conversation between a human and an AI assistant called `CHATBOTNAME`, developed by `DEVELOPER`.")와 같은 prefix를 넣어주는 것도 방법

이 외에도 max length를 채우기 위해 여러 데이터를 하나의 시퀀스로 concat하는 방법도 존재한다.

### The Effect of Instruction Tuning

그렇다면 instruction tuning의 효과는 어떠할지 살펴보자.

#### Performance Improvement

비록 보통의 양으로 학습시켰다 할지라도 instruction tuning은 LLM의 성능을 향상시키거나 잠금해제하는데 중요한 방법으로 자리잡아가고 있다.
77M - 540B의 다양한 scale의 LLM에서 전반적인 성능 향상을 보인 연구들이 많으며 \[[69][69], [345][345]\], 파라미터가 증가할수록 좋은 성능을 보인다.
더욱이 sLLM에 instruction tuning을 하는 경우 fine-tuning을 진행하지 않은 LLM보다 성능이 더 좋은 경우도 있다 \[[28][28], [69][69]\].
Model scale과 마찬가지로 instruction tuning 또한 다양한 아키텍처, objective, model adaptation에서 일관적인 성능 향상을 보이고 있다.

실전에서 instruction tuning은 sLLM, LLM 모두에서 능력을 향상시키는 일반적인 접근법이라 볼 수 있으며, LLM이 필요한 instruction data의 양이 pre-training의 데이터보다 훨씬 적으므로 더 경제적이라 할 수 있겠다.

#### Task Generalization

Instruction tuning은 특정 task를 수행하는데 필요한 natural language instruction을 모델에게 이해하게끔 만든다.
이는 LLM에게 능력(=emergent ability)을 부여하여 demonstration 없이, 심지어는 한번도 보지 못한 task에서도 사람의 지시사항에 따라 특정 task를 수행하도록 만든다.
이는 다양한 연구에서 확인된 것으로, unseen/seen task에 대해 우월한 성능을 보인바 있다 \[[95 (OPT-IML)][95], [345][345]\].

또한, instruction tuning은 LLM의 몇몇 약점들을 (e.g., 특정 작업을 수행하지 않고 반복적으로 생성하거나 input을 보완하는 것) 완화하며, cross lingual에서도 일반화하는 성능을 보였다.
예를 들어 [[BLOOMZ-P3][94]\]에 대해 영어 task인 P3 데이터를 학습시켰을 때, 여러 언어로 이루어진 sentence completion task에서도 50% 이상의 성능향상이 발생하였다.
이는 영어로만 이루어진 데이터셋에서도 일반적인 task 스킬을 익힐 수 있으며, 이러한 스킬을 다른 언어로 transfer하는 것으로 보인다.
뿐만 아니라 영어로만 이루어진 instruction을 통해 multilingual task에서 만족스러운 결과를 생성할 수 있었으며, 특정 언어로의 instruction enginerring에 대한 노력을 줄일 수 있다.

#### Domain Specialization

LLM이 좋은 성능을 보이더라도 의약, 법, 금융과 같은 domain knowledge는 부족할 수 있다.
Instruction tuning은 이러한 문제에도 효과적인 방법이다.

대표적으로 Flan-PaLM을 활용한 의학 전문 LLM인 Med-PaLM, FLAN-T5을 활용한 추천 시스템 등이 있다.

### Empirical Analysis for Instruction Tuning

다양한 instruction dataset을 사용하면 downstream task에서 성능이 다양해지는 결과가 나온다.
여기서는 LLaMA 7B와 *13B*를 사용하여 instruction type에 따른 영향을 살펴보도록 한다.

13B는 리소스 문제로 진행하지 않음.
{: .notice--info}

#### Instruction Datasets

[Formatted Instance Construction](#Formatted-Instance-Construction)에서 살펴본 것과 같이 다음의 세 가지 일반적인 형태의 instruction을 본 실험에서 고려하였다.

- Task-specific instruction: 가장 일반적으로 사용되는 multi-task instruction dataset(e.g. FLAN-T5 \[[69][69]\])을 적용하였다. 이에는 1836개의 task와 15M 이상의 instruction을 포함한다.
- Daily chat instruction: 사용자가 일상생활에서 사용한 것으로 real-life scenario에 적합하다. 여기서는 63K개의 실제 사용자가 사용한 instruction인 ShareGPT의 데이터셋을 사용하였다. 이는 Vicuna에서 사용한 핵심 instruction이기도 하다.
- Synthetic instruction: LLM을 활용하여 자동으로 생성한 인공 데이터셋도 고려하였는데, 82K의 instance input/output에 대한 52K개의 instruction으로 이루어진 Self-Instruct-52K \[[143][143]\]를 사용하였다.

FLAN-T5의 경우는 매우 크므로 (15M 이상) 이 중 80K개만 샘플링하여 다른 데이터셋과의 공평한 비교를 진행하였다.

#### Improvement Strategies

TODO: PEFT 내용이 급하므로 중간 스킵


## Parameter-Efficient Model Adaptation

이제는 LLM을 효율적으로 학습하는 PEFT method를 살펴보자.
우선 Transformer language model을 이용한 PEFT 방식을 살펴보고 이에 대해 요약하도록 하겠다.

### Parameter-Efficient Fine-Tuning Methods

PEFT\[[145 (LoRA)][145], [396 (Prefix-Tuning)][396], [397 (Prompt Tuning)][397]\]는 성능은 유지하되 trainable parameter의 수는 감소시키는 기법이다.
다음을 통해 adapter tuning, prefix tuning, prompt tuning, LoRA를 살펴보도록 하겠다.

TODO: adapter, prefix, prompt 할 것

#### Low-Rank Adaptation (LoRA)

LoRA는 각 레이어의 update matrix를 근사하기 위해 low-rank constraint를 걸어 학습 시 필요한 파라미터의 양을 줄인다.

모델의 웨이트 업데이트 프로세스는 $\mathrm{W} \gets \mathrm{W}+\Delta \mathrm{W}$로 표현할 수 있다.
LoRA는 LLM의 웨이트 $\mathrm{W} \in \mathbb{R}^{m \times n}$는 고정시킨채로 parameter update $\Delta \mathrm{W}$를 low-rank decomposition matrix $\Delta \mathrm{W} = \mathrm{A} \cdot \mathrm{B} ^{\intercal}$ where $\mathbb{A}^{m \times k}$ and $\mathbb{B}^{n \times k}$로 근사시킨다. $k \ll \min(m, n)$는 reduced rank이다.

몇몇 연구에서는 적절한 rank를 선택하는 principled approach를 제안하기도 하였다 (importance score based allocation ([AdaLoRA][406]), search-free optimal rank selection ([Dylora][407]).


#### Parameter-Efficient Fine-Tuning on LLMs

LLM이야말로 PEFT의 효과를 가장 많이 볼 수 있는 영역이다.
LoRA는 LLaMa, BLOOOM과 같은 오픈 소스 LLM에 가장 널리 쓰이는 기법이며, 이 중 특히 LLaMa가 가장 큰 관심을 받았다.

Alpaca-LoRA는 Alpaca의 경량화된 tuning 버전으로 7B LLaMA에 52K로 이루어진 instruction을 따르는 *human demonstration*으로 이루어져 있다.

Alpaca나 Alpaca-LoRA 모두 human demonstration과는 관계가 없는 것으로 보이는데, 저자의 오류인지 정확히 확인하기 어렵다.
{: .notice--warning}

또 LLaMA-Adapter \[[409][409]\]에서는 각 Transformer 레이어에 learnable prompt vector를 삽입하는데, 여기서는 zero-initialized attention을 제안하여 
under-fitted prompt vector의 영향을 완화, 이로 인해 학습을 상승시킬 수 있었다.

\[[399 (llm-adapters)][399]\]에선 다양한 tuning method이 LM에 미치는 영향에 대해 mpirical study를 수행하였다.
4개의 PEFT (adapter tuning, parallel adapter tuning [400, 410], LoRA)를 GPT-J (6B), BLOOM (7.1B), LLaMA (7B)에 적용하였고, math reasoning dataset으로 평가한 결과 간략한 task의 경우 baseline인 GPT-3.5와 비교할만한 성능을 내었으나, 어려운 task의 경우 baseline보다 낮은 성능을 내었다.
전체적으로 LoRA의 경우 다른 방법론보다 훨씬 더 낮은 trainable parameter로 더 좋은 성능을 내었다.

현재 사용가능한 리소스로는 PEFT 라이브러리가 있으며, 사용 가능한 방법론은  LoRA/AdaLoRA, prefix-tuning, P-Tuning, prompt-tuning이 있다.

[Parameter-Efficient Fine-Tuning Methods](#Parameter-Efficient-Fine-Tuning-Methods)에서 논의한바와 같이 다양한 PEFT 방법론이 있으나, 대부분은 sLLM에서 실험한 결과이며, 지금까지는 다양한 setting/task에서 LLM에 대한 PEFT의 영향에 대한 철저한 조사는 여전히 부족한 상황이다.





![Fig.1-add-caption-here]({{site.url}}{{site.baseurl}}/assets/posts/nlp/){: .align-center}{: width="600"}

![Caption](URL){: .align-center}{: width="600"}


`{: .notice--info}`는 추가적인 정보를 적을 때 사용한다.
{: .notice--info}

`{: .notice--warning}`는 추후 살펴볼 내용을 적는다.
{: .notice--warning}

`{: .notice--warning}`는 잘 모르는 것을 적는다.
{: .notice--warning}

\[[348][348]\]

[28]: https://arxiv.org/abs/2110.08207
[67]: https://arxiv.org/abs/2109.01652
[69]: https://arxiv.org/abs/2210.11416
[88]: https://aclanthology.org/2022.emnlp-main.340/
[94]: https://aclanthology.org/2023.acl-long.891/
[95]: https://arxiv.org/abs/2212.12017
[143]: https://arxiv.org/abs/2212.10560
[145]: https://arxiv.org/abs/2106.09685
[166]: https://aclanthology.org/2022.acl-long.244/
[168]: https://aclanthology.org/2023.findings-acl.558/
[342]: https://arxiv.org/abs/2303.10475
[345]: https://arxiv.org/abs/2301.13688
[348]: https://arxiv.org/abs/2308.06259
[349]: https://arxiv.org/abs/2305.11206
[350]: https://arxiv.org/abs/2307.08701
[351]: https://arxiv.org/abs/2306.02707
[352]: https://github.com/RUC-GSAI/YuLan-Chat
[353]: https://arxiv.org/abs/2306.04751
[396]: https://aclanthology.org/2021.acl-long.353/
[397]: https://aclanthology.org/2021.emnlp-main.243/
[399]: https://aclanthology.org/2023.emnlp-main.319/
[406]: https://arxiv.org/abs/2303.10512
[407]: https://arxiv.org/abs/2303.10512
[409]: https://arxiv.org/abs/2303.16199