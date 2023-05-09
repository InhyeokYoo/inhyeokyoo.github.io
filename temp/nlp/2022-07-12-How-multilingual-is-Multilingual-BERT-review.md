---
title:  "[작성 중] How multilingual is Multilingual BERT review"
toc: true
toc_sticky: true
permalink: /project/nlp/How-multilingual-is-Multilingual-BERT/
categories:
  - NLP
  - Paper Review
tags:
  - Language Modeling
use_math: true
last_modified_at: 2022-07-12
---

## Introduction

본 논문은 ACL 19년도

- [논문 보러가기](https://aclanthology.org/P19-1493/)


## Challenging

ELMo, BERT와 같은 deep, contextualized language models은 대량의 코퍼스로부터 pre-train되고, 특정한 downstream task에 fine-tuning되는 식으로 학습한다.
이전의 모델 탐구(model probing)에 관련된 연구들은 이러한 모델의 representation이 여러 정보들 중 특히 문장(통사, syntatic)과 개체명(named entity) 정보를 인코딩할 수 있음을 보여왔지만, 여태까지는 영어로 학습한 모델이 영어로 된 정보를 학습하는 것에만 국한되어 있다.

나무위키에서는 통사론에 대해 다음과 같이 설명하고 있다.

>  **문장과 구의 구조를 공부**하는 언어학의 분야. 의사소통 상의 의미와 독립된(즉, 언어적으로만 고려되는) 문장의 형식 그 자체를 해부하며, 비문과 정문을 판별하고, 문장을 생성 및 분해하는 규칙을 연구한다. 정상인이라면 적어도 1개 이상의 언어를 사용한다. 그리고 본인이 모국어로 사용하는 언어에 대해서는 직관을 가진다. 직관은, 특정 언어표현이 문법적인 문장(정문)인지 아닌지(비문)인지 판단하는 능력이자, 어떤 문장이 특정 의미를 가지는지 아닌지 여부를 판단하는 능력이다. 통사론의 연구대상은 바로 이 '문법적 직관'이다.

## Contributions

따라서 본 논문에서는 영어에 국한하는 것이 아닌 LM의 표현이 언어를 넘나들며 일반화하는 정도에 대해 empirically 탐구한다.
이는 **104개의 위키피디아 단일 코퍼스로부터 학습한 Multilingual BERT(이후 M-BERT)**를 하나의 language model로 이용하여 진행한다.

M-BERT는 zero-shot cross-lingual model transfer에 매우 간단하게 접근이 가능하기 때문에 probing study에 매우 적합하다고 볼 수 있다.
본 논문에서는 **하나의 언어에 대한 특정 태스크를 fine-tuning한 후, 다른 언어를 통해 이를 평가**할 것이다.
따라서 모델이 언어를 넘나들며 정보를 일반화하는 방법을 관찰할 수 있다.

이는 기존 BERT의 [multilingual.md](https://github.com/google-research/bert/blob/master/multilingual.md)에서 소개하는 방법과도 비슷한데, MultiNLI 를 여러 언어로 번역한 XNLI 데이터셋을 이용하여 다른 언어에서의 NLI 능력을 평가하는 것이다.
그리고 놀랍게도, 꽤나 잘 동작한다.

## Method

### Models and Data

모델의 경우는 BERT와 거의 동일하다.
대신 앞서 설명했던 것과 같이 monolingual English data에만 학습한 것이 아닌 104개의 위키피디아 코퍼스도 같이 학습했기 때문에 사전을 공유한다.
기존 BERT와 동일하게 특정 인풋을 통해 모델에 특정 언어의 정보를 알려주거나, objective를 변경하는 등의 작업도 하지 않는다.
그냥 BERT와 완전히 동일한 방법으로 학습한다.

NER의 경우에는 네덜란드어, 스페인어, 영어, 독어로 이루어진 CoNLL-2002/2003를 활용했으며, 이에 추가로 구글에서 제공하는(in-house dataset) 16개의 언어로 이루어진 데이터를 이용한다.
구성방식은 CoNLL과 동일하다.

아래는 이에 대한 결과이다.

![image](https://user-images.githubusercontent.com/47516855/178471104-341b20a8-cb8e-4f1f-8859-c25759ce26ee.png){: .align-center}{: width="300"}

Part of Speech의 경우에는 41개의 언어에 대한 Universal Dependencies (UD)를 사용한다.

![image](https://user-images.githubusercontent.com/47516855/178514609-76a7d053-291d-479d-b9ff-1bfce4f7ac3a.png){: .align-center}{: width="300"}

아래 표는 POS에 대한 결과이다.


![image](https://user-images.githubusercontent.com/47516855/178471236-cf435105-538f-4807-8926-67e6920960e9.png){: .align-center}{: width="300"}

## Vocabulary Memorization

이제 본격적으로 실험에 대한 내용이다.
M-BERT는 동일한 사전을 사용하기 때문에, 학습에 나타났던 word piece가 테스트에도 나타나 성능 평가에 영향을 미칠 수 있다.
따라서 피상적인 형태(superficial form)의 일반화 능력을 살펴보도록 한다.
본 섹션에서 탐구할 내용은 다음과 같다:
- transferability가 **어휘적 중복(lexical overlap)에 얼마나 영향**을 받는가?
- 다른 문자(script)로 쓰인, 즉, **중복이 없는 경우에도 transfer가 가능한가?**

### Effect of vocabulary overlap

만일 M-BERT의 성능이 전적으로 vocabulary memorization에 의존하고 있다면, zero-shot에서의 성능 또한 word piece overlap에 의존할 것이다.
이를 확인하기 위해 overlap을 학습 시에 사용했던 word piece와 평가 시 사용한 word piece 간의 교집합과 합집합의 비율로 정의하여 이를 확인하였다.


![image](https://user-images.githubusercontent.com/47516855/178473370-2bcfe64d-5584-43cb-92c5-be5cb4e7176f.png){: .align-center}{: width="300"}

위 그림은 16개 언어에 대한 in-house dataset를 M-BERT와 영어로만 학습한 EN-BERT로 실험한 결과이다.
CoNLL의 4개의 데이터로도 실험해보았지만, 위 그림과 비슷한 trend를 보임을 확인하였다고 한다.

영어로만 학습한 EN-BERT의 경우 중복이 감소할 수록 transfer 성능이 약화되는 것을 확인하였다.
또한, 다른 문자로 쓰여 중복이 거의 없는 경우 성능이 거의 0에 가까운 것을 볼 수 있다.

반면 M-BERT의 경우 중복에 대해 좀 더 평평한 형태로 그래프가 나타났고, 중복이 거의 없는 경우에도 40%에서 70%사이의 성능을 보였다.
이를 통해 **단순 vocabulary memorization를 넘어선 representational capacity**를 갖는 것으로 보인다.

EN-BERT가 이러한 결과를 보인 것이 word piece로 다른 문자를 표현하는 것이 불가능해서인지, 아니면 명확하게 multilingual representation이 부족해서인지 확인하기 위해 추가로 non-cross-lingual setting에서 실험을 진행하였다.

![image](https://user-images.githubusercontent.com/47516855/178518443-f82939a9-ba67-4097-a14f-eb3a5cfb161e.png){: .align-center}{: width="300"}

EN-BERT도 다른 언어에 대해 fine-tuning을 진행하였을 경우 성능이 이전 SOTA와 맞먹는 것을 보아 word piece의 문제가 아닌, multilingual representation이 부족한 것으로 기인하는 문제임을 확인하였다.


### Generalization across scripts




This provides clear evidence of M-BERT’s multilingual representation
ability, mapping structures onto new vocabularies
based on a shared representation induced solely
from monolingual language model training data.

However, cross-script transfer is less accurate
for other pairs, such as English and Japanese, indi-
cating that M-BERT’s multilingual representation
is not able to generalize equally well in all cases.

A possible explanation for this, as we will see in
section 4.2, is typological similarity. English and
Japanese have a different order of subject, verb
and object, while English and Bulgarian have the
same, and M-BERT may be having trouble gener-
alizing across different orderings

## Encoding Linguistic Structure

we present probing experiments that
investigate the nature of that representation: 
How
does typological similarity affect M-BERT’s abil-
ity to generalize? 
Can M-BERT generalize from
monolingual inputs to *code-switching* text? 
Can
the model generalize to transliterated text without
transliterated language model pretraining?

> 언어유형론(言語類型論, 영어: linguistic typology) 또는 단순히 유형론은 언어학에서 단순하게는 세계 여러 언어들을 조사하여 그 유형을 분류하는 연구를 말한다. 더 나아가, 유형론은 단순한 조사와 분류에서 끝나는 것이 아니라, 이를 일반화하여 인간의 언어가 가지는 보편적인 성격을 탐구하는 것을 말한다.

언어학의 하나의 방법론으로써 유형론은 이러한 ‘언어의 유형을 연구하는 것’만을 의미하지는 않는다. 유형론적 연구는 형식적 혹은 논리적 연구에 맞서는 것으로 언어의 기능, 인식 구조, 화용적 성격, 역사적 성격을 중요시여기는 방법론을 말한다. 다시 말해, 형식적 문법 연구가 이론내적 개념을 기반으로 하며 언어 외적 요소를 배제하려는 것인데 비하여, 유형론은 언어 외적인 실세계의 문제를 적극적으로 끌어 들여 언어 현상을 설명하려는 방법론이다. 


### Effect of language similarity

performance improves with similar-
ity, showing that it is easier for M-BERT to map
linguistic structures when they are more similar,
although it still does a decent job for low similar-
ity languages when compared to EN-BERT.

### Generalizing across typological features

Table 5 shows macro-averaged POS accuracies for
transfer between languages grouped according to
two typological features: subject/object/verb or-
der, and adjective/noun order

The results reported include only
zero-shot transfer, i.e. they do not include cases
training and testing on the same language.

performance is best when transferring
between languages that share word order features,
suggesting that while M-BERT’s multilingual rep-
resentation is able to map learned structures onto
new vocabularies, it does not seem to learn sys-
tematic transformations of those structures to ac-
commodate a target language with different word
order.

### Code switching and transliteration

Code-switching (CS): the mixing of multiple languages within a single utterance
https://en.wikipedia.org/wiki/Code-switching

transliteration: writing that is not in the language’s standard script

transliteration은 원문의 철자를 하나하나 다른 문자 체계로 옮겨 적는 '전자법'을 말하고(예: 독립문 Doglibmun)
https://en.wikipedia.org/wiki/Transliteration


CS/transliteration present unique test cases for M-BERT, which is pre-trained on monolingual, standard-script corpora.

Generalizing to code-switching is similar to other cross-lingual trans-
fer scenarios, but would benefit to an even larger
degree from a shared multilingual representation.

Likewise, generalizing to transliterated text is sim-
ilar to other cross-script transfer experiments, but
has the additional caveat that M-BERT was not
pre-trained on text that looks like the target.

We test M-BERT on the CS Hindi/English UD
corpus from Bhat et al. (2018), which provides
texts in two formats: *transliterated*, where Hindi
words are written in Latin script, and *corrected*,
where annotators have converted them back to De-
vanagari script. 

Table 6 shows the results for models fine-tuned using a combination of monolingual
Hindi and English, and using the CS training set
(both fine-tuning on the script-corrected version of
the corpus as well as the transliterated version).

For script-corrected inputs, i.e., when Hindi
is written in Devanagari, M-BERT’s performance
when trained only on monolingual corpora is com-
parable to performance when training on code-
switched data, and it is likely that some of the
remaining difference is due to domain mismatch.
This provides further evidence that M-BERT uses
a representation that is able to incorporate infor-
mation from multiple languages.

However, M-BERT is not able to effectively
transfer to a transliterated target, suggesting that
it is the language model pre-training on a particu-
lar language that allows transfer to that language.

M-BERT is outperformed by previous work in
both the monolingual-only and code-switched su-
pervision scenarios. Neither Ball and Garrette
(2018) nor Bhat et al. (2018) use contextualized
word embeddings, but both incorporate explicit
transliteration signals into their approaches.


## Multilingual characterization of the feature space

In this section, we study the structure of
M-BERT’s feature space. If it is multilingual, then
the transformation mapping between the same
sentence in 2 languages should not depend on the
sentence itself, just on the language pair.

We sample 5000 pairs of sentences from WMT16
(Bojar et al., 2016) and feed each sentence (sep-
arately) to M-BERT with no fine-tuning.

We
then extract the hidden feature activations at each
layer for each of the sentences, and average the
representations for the input tokens except [CLS]
and [SEP], to get a vector for each sentence, at
each layer

For each pair of sentences, we compute the vector point-
ing from one to the other and average it over all
pairs:

Our intuition is that the lower layers have more “token
level” information, which is more language dependent, par-
ticularly for languages that share few word pieces.

t achieves over 50%
accuracy for all but the bottom layers,9 which
seems to imply that the hidden representations, al-
though separated in space, share a common sub-
space that represents useful linguistic information,
in a language-agnostic way.

Similar curves are ob-
tained for EN-RU, and UR-HI (in-house dataset),
showing this works for multiple languages.

As to the reason why the accuracy goes down in
the last few layers, one possible explanation is that
since the model was pre-trained for language mod-
eling, it might need more language-specific infor-
mation to correctly predict the missing word




## Summary

{: .align-center}{: width="300"}
