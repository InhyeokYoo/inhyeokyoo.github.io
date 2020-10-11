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



