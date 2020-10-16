---
title:  "Minimal Mistakes: github pages를 잘 꾸며모자"
excerpt: "Minimal Mistakes를 내 입맛대로 수정하기"
toc: true
toc_sticky: true

categories:
  - Github Pages
tags:
  - jekyll
  - minimal mistakes
use_math: true
last_modified_at: 2020-10-11
---

내 입맛대로 Minimal mistakes를 수정하는 과정에서 생기는 문제와 해결방안을 정리해보았다.

# TOC 폰트 사이즈 수정

TOC 폰트 사이즈에 대한 지정은 `_sass\minimal-mistakes\_navigation.scss` 파일에서 할 수 있다.
이를 보면 다음과 같은 항목이 있는데,

```scss
.toc__menu {
  margin: 0;
  padding: 0;
  width: 100%;
  list-style: none;
  font-size: $type-size-6;

  @include breakpoint($large) {
    font-size: 0.8em; //$type-size-6;
  }
```

여기서 breakpoint의 `font-size`를 적절하게 수정해주면 된다. 각 type-size는 `_sass\minimal-mistakes\_variables.scss`에 다음과 같이 할당되어 있다.

```scss
/* type scale */
$type-size-1: 2.441em !default; // ~39.056px
$type-size-2: 1.953em !default; // ~31.248px
$type-size-3: 1.563em !default; // ~25.008px
$type-size-4: 1.25em !default; // ~20px
$type-size-5: 1em !default; // ~16px
$type-size-6: 0.75em !default; // ~12px
$type-size-7: 0.6875em !default; // ~11px
$type-size-8: 0.625em !default; // ~10px
```

개인적으로는 5는 너무 크고, 6은 너무 작아서 `0.8em`값으로 따로 주었다.

# Notice 사용하기

다음과 같은 **notice**를 사용할 수 있다. 사용법은 문단의 끝에 `{: .notice}`를 사용하는 것이다. 

**서비스 변경:** 서비스 변경과 같이 간단한 안내는 `{: .notice}`를 문단의 끝에 첨부함으로서 사용할 수 있다.
{: .notice}

**중요한 노트:** 조금 더 중요한 문구는 `{: .notice--primary}`를 통해 사용할 수 있다.
{: .notice--primary}

<div class="notice--primary" markdown="1">
**중요한 노트와 코드블록:** 조금 더 중요한 문구와 함께 코드를 사용하는 것은 아래 코드블록을 통해 사용할 수 있다.

```html
<div class="notice--primary" markdown="1">
**중요한 노트와 코드블록:** 조금 더 중요한 문구와 함께 코드를 사용하는 것은 아래 코드블록을 통해 사용할 수 있다.

// 이하 코드블록
```
</div>

**정보 안내:** 정보 안내는 `{: .notice--info}`를 이용한다.
{: .notice--info}

**경고 안내:** 경고 안내는 `{: .notice--warning}`를 이용한다.
{: .notice--warning}

*위험 안내:** 위험 안내는 `{: .notice--danger}`를 이용한다.
{: .notice--danger}

**성공 안내:** 성공 안내는 `{: .notice--success}`를 이용한다.
{: .notice--success}