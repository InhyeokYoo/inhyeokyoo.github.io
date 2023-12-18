---
title:  "RAG: Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks review"
toc: true
toc_sticky: true
permalink: /project/nlp/review/rag/
categories:
  - NLP
  - Paper Review
tags:
  - Open QA
use_math: true
last_modified_at: 2022-05-31
---

## Introduction

간략한 개요와 참고 리소스 등을 적는다.

- Parameterizing : 모델의 가중치에 지식을 주입하는 과정입니다. 우리가 다양한 목적함수를 바탕으로 Large Model을 Pre-training하는 이유가 결국 knowledge를 parameterizing하기 위함이라고 볼 수 있죠.
- Knowledge Intensive Tasks : 사람조차도 외부지식 (ex. 위키피디아 검색) 없이 해결하기 어려운 문제를 일컫습니다. 즉 모델의 관점에서 보면, parameterized되지 못한 외부 지식이 필요한 문제입니다.
- MIPS : Maximum Inner Product Search의 약어로, 우리에게 vector space에 mapping된 query $x$가 있고 여러 외부 정보들 $d _i$가 있다고 가정할 때 query $x$와 내적(or 코사인 유사도)가 높은 외부 정보들 $d _i$를 찾는 과정을 의미합니다. 최근의 Facebook의 FAISS가 이를 빠르게 구현해놓은 좋은 라이브러리로 각광받고 있습니다.

### Summary

전체 내용을 요약한다

- 문제는 뭐였고
- 이렇게 해서
- 저렇게 해결한다
- 결론은 어떻게 나온다.

## Challenges

- ODQA: question이 주어졌을 때 knowledge base에서 retrieval를 통해 얻은 passage에서 reader가 answer를 추론하는 task (정답 토큰 존재)
- Fact checking: Knowledge Intensive Task (KIT) 중 하나로, ODQA와 비슷하게 passage를 불러오지만, 주 목적은 주어진 passage를 통해 해당 문장이 진실인지 판별 (정답 토큰 존재하지 않음)

데이터:
- Question: 질문으로 외부 지식 없이 쉽게 답할 수 없음.
- Answer: passage 내에서 연속된 span으로 존재한다고 보장 불가
- Knowledge Base: 외부 지식 문서 (e.g. 위키피디아)
- Passage: 질문과 관련된 문서로 knowledge Base에서 선택



## Contributions

주요 contribution을 요약하여 적는다.

Parametrized Implict Knowledge Base

## Related Work

- DrQA: 

## Method

$$
\begin{align}
\max P(a \lvert q) &= \max P(a, p \lvert q) \\
&= \underbrace{\max P(a \lvert p, q)} _{\text{Reader}} \underbrace{P(p \lvert q)} _{\text{Retrieval}}
\end{align}
$$

- Reader: Question과 passage가 인풋일 때 answer의 likelihood가 최대가 되도록 학습
- Retrieval: Question이 인풋일 때 사용할 passage의 likelihood를 최대화하도록 학습


### Detail-Method

구체적인 프로세스를 적는다.
제목은 논문에서 따온다

## Experiment

간략하게 실험에 대한 overview를 적는다

### Detail-Experiment

자세한 프로세스를 적는다.
제목은 논문에서 따온다.

## Conclusion

논문의 결론과 개인적인 소감, 아쉬운 점, 응용법 등을 정리한다.

![Fig.1-add-caption-here]({{site.url}}{{site.baseurl}}/assets/posts/CATEGORY/POST-NAME-Fig.1.png){: .align-center}{: width="600"}

![Caption](URL){: .align-center}{: width="600"}

