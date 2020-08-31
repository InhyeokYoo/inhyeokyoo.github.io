---
title:  "Markdown 수식 정리"
excerpt: "혼자서 정리하는 markdown 수식"
toc: true
toc_sticky: true

categories:
  - IT

use_math: true
last_modified_at: 2020-08-31
---

# Intro

Markdown을 통해 Github pages를 작성하다 보니 수식을 입력하는게 생각보다 까다롭다. Colab에선 이러지 않았는데... 따라서 직접 사용하고 정리해보는 Markdown 수식을 작성해보았다. 본 post는 기본적으로 markdown에 대한 이해가 어느정도 있는 상태에서 사용할 것을 권장한다. 기본적인 자료는 구글링하면 쉽게 구할 수 있으니 구해보자.

# 기본

## 수식 정렬

`\begin{align}`과 `\end{align}`을 사용하면 **정렬이 오른쪽**으로 된다. 따라서 `&`를 통해 어디서부터 시작할지 정해주면 잘 작동하게 된다. 다음은 예시이다.

**오른쪽 정렬**
$$
\begin{align}
R _k = \{ x^LM _k, \overrightarrow {h^{LM} _{k, j}}, \overleftarrow {h^{LM} _{k, j}} \lvert j=1, ..., L\} \\
= \{ h^{LM} _{k, j} \lvert j=0, ..., L \},
\end{align}
$$

```
\begin{align}
R _k = \{ x^LM _k, \overrightarrow {h^{LM} _{k, j}}, \overleftarrow {h^{LM} _{k, j}} \lvert j=1, ..., L\} \\
= \{ h^{LM} _{k, j} \lvert j=0, ..., L \},
\end{align}
```

---

**왼쪽 정렬**

$$
\begin{align}
R _k &= \{ x^LM _k, \overrightarrow {h^{LM} _{k, j}}, \overleftarrow {h^{LM} _{k, j}} \lvert j=1, ..., L\} \\
&= \{ h^{LM} _{k, j} \lvert j=0, ..., L \},
\end{align}
$$

```
$$
\begin{align}
R _k &= \{ x^LM _k, \overrightarrow {h^{LM} _{k, j}}, \overleftarrow {h^{LM} _{k, j}} \lvert j=1, ..., L\} \\
&= \{ h^{LM} _{k, j} \lvert j=0, ..., L \},
\end{align}
$$
```

보다시피 $R _k$옆에 &을 넣어줘서 다음부터는 $=$부터 정렬이 되게끔 만들었다.



# Greek Letter

escape sequence에 알파벳을 써주면 작성할 수 있다. 

| Greek Letter | Markdown |
| :---: | :---: |
| $\alpha$ | \alpha |
| $\theta$ | \theta |

# Vector

`\vec`

`\mathbf`

