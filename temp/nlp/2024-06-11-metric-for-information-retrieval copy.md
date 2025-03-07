---
title:  "정보검색(Information Retrieval) 평가지표"
toc: true
toc_sticky: true
categories:
  - Information Retrieval
use_math: true
last_modified_at: 2024-06-11
---

## 들어가며

https://www.evidentlyai.com/ranking-metrics/evaluating-recommender-systems#what-is-a-recommender-system

https://www.evidentlyai.com/ranking-metrics/mean-reciprocal-rank-mrr

- Recommendations and ranking systems share the goal of returning a list of items sorted by relevance.
- To evaluate recommendations, you must have predictions as user-item pairs, the binary or graded relevance score as the ground truth, and choose the K parameter.
- To measure the actual impact, you can track business metrics like sales, click-through rates, or conversions during online evaluations.

## 

A recommender system solves a ranking task. It returns a list of sorted items that might be relevant for a specific user. Other ranking examples include search and information retrieval, where the goal is to rank documents relevant to a particular query.

The recommender system is one flavor of a ranking task, but there are others. Ultimately, the goal of any ranking system is to help prioritize and arrange items in a list. However, the expectations on how to do it best may vary.

A different example of a ranking task is an information retrieval system like an internet search engine. Unlike recommendations where you want to find items that broadly match a “user’s profile,” search happens after an explicit user’s input or query.

Think about the last time you typed something in a search engine. Chances are, you did not look for it to surprise or entertain you with unexpected links. Instead, you wanted to find the correct response to your query at the very top of the page.

Despite their differences, the output of any ranking system usually looks the same: it is a **list** of items (movies, products, items, documents, links) ordered by their expected relevance. Since all ranking systems aim to present items in a sorted manner, you can often use the same metrics to evaluate them. Of course, there are a few nuances regarding when each metric fits best for each task. 



## 데이터

IR 평가를 위해 데이터에 있어야 할 것은 다음과 같다:
- Prediction: 모델이 평가한 예측치.
- Ground truth: 실제 연관 (relevance) 여부 혹은 점수

여기서 관련성(relevance)은 리스트 내 아이템 개별의 품질을 반영하는 것으로, binary (e.g., clicking, watching, buying) 혹은 graded score (raing from 1 to 5)로 구성된다.
검색엔진의 경우 주어진 query에 대한 정답을 포함하는 문서가 될수도 있고, 온라인 스토어의 경우 고객이 사고싶은 물건일 수도 있다.






{: .align-center}{: width="300"}
