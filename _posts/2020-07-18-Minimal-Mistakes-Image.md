---
title:  "고급 Minimal Mistakes: 이미지편"
excerpt: "minimal mistakes 이미지를 다뤄보자."
toc: true
toc_sticky: true

categories:
  - IT
tags:
  - github
  - github pages
  - github desktop
  - jekyll
  - minimal mistakes
use_math: true
last_modified_at: 2020-07-18
---

Github pages를 만들면서 나름 재미를 느껴 하던거 내팽개치고 이것저것 변경해보았지만 어려운 것이 너무 많았다. 이번 포스트에선 mminimal 템플릿을 이것저것 고치는 법을 알아보겠다.

# Image

## Image alignment

Minimal Mistakes [Uitlity Classes](https://mmistakes.github.io/minimal-mistakes/docs/utility-classes/#image-alignment)에는 이미지를 정렬하는 유용한 기능을 제공한다.

```markdown
![image-right](/assets/images/filename.jpg){: .align-right}
![image-right](/assets/images/filename.jpg){: .align-left}
![image-right](/assets/images/filename.jpg){: .align-center}
```

**가운데 정렬**을 하면 다음과 같이 된다.  

![가운데 이미지](https://mmistakes.github.io/minimal-mistakes/assets/images/image-alignment-580x300.jpg){: .align-center}

---

![우측 이미지](https://mmistakes.github.io/minimal-mistakes/assets/images/image-alignment-150x150.jpg){: .align-right} 이 문서의 **우측 정렬 결과**는 다음과 같다. 보다시피 오른쪽으로 가게 된다. 이 문단의 텍스트는 자동으로 그림 왼쪽에 정렬된다. 그러나 주의할 점이 있는데, 아래와 위, 그리고 옆에 충분한 공간이 있어야 한다는 점이다. 만일 그렇지 않으면 레이아웃이 깨지게 된다. 여기서부터는 충분한 공간을 채우기 위해 막 지껄였다. 그래야지만 레이아웃이 안 깨지니까. 그냥 계속 말하는 중이다. 아직도 안 채워졌다. 여전히 안 채워졌다. 이제 됐나? 됐다. 

---


![좌측 이미지](https://mmistakes.github.io/minimal-mistakes/assets/images/image-alignment-150x150.jpg){: .align-left} 이 문서의 **좌측 정렬 결과**는 다음과 같다. 보다시피 왼쪽으로 가게 된다. 이 문단의 텍스트는 자동으로 그림 왼쪽에 정렬된다. 그러나 주의할 점이 있는데, 아래와 위, 그리고 옆에 충분한 공간이 있어야 한다는 점이다. 만일 그렇지 않으면 레이아웃이 깨지게 된다. 여기서부터는 충분한 공간을 채우기 위해 막 지껄였다. 그래야지만 레이아웃이 안 깨지니까. 그냥 계속 말하는 중이다. 아직도 안 채워졌다. 여전히 안 채워졌다. 이제 됐나? 됐다. 

## Image size

html 사용하거나,
```html
<img src=imgurl width=300 height=500>
```

markdown을 이용한다.
```markdown
![image-name](image-url){: width="400" height="200"}
```

아래는 원본 이미지와 사이즈가 변경 된 이미지다. 둘 다 이쁘라고 가운데 정렬을 해주었다.

![큰 이미지](https://mmistakes.github.io/minimal-mistakes/assets/images/image-alignment-580x300.jpg){: .align-center}


<img src='https://mmistakes.github.io/minimal-mistakes/assets/images/image-alignment-580x300.jpg' width=300>{: .align-center}

## Markup

[마크업](https://mmistakes.github.io/minimal-mistakes/markup/markup-image-alignment/)을 통해서도 진행할 수 있다. 이 경우 장점은 정렬과 크기조절, figcaption을 동시에 할 수 있다는 점이다.

<figure class="align-left">
  <img src='https://mmistakes.github.io/minimal-mistakes/assets/images/image-alignment-150x150.jpg' alt="">
  <figcaption>왼쪽 정렬한 모습</figcaption>
</figure>
위에서 진행한 것을 다시 한 번 해보자. 이 문서의 **좌측 정렬 결과**는 다음과 같다. 보다시피 왼쪽으로 가게 된다. 이 문단의 텍스트는 자동으로 그림 왼쪽에 정렬된다. 그러나 주의할 점이 있는데, 아래와 위, 그리고 옆에 충분한 공간이 있어야 한다는 점이다. 만일 그렇지 않으면 레이아웃이 깨지게 된다. 여기서부터는 충분한 공간을 채우기 위해 막 지껄였다. 그래야지만 레이아웃이 안 깨지니까. 그냥 계속 말하는 중이다. 아직도 안 채워졌다. 여전히 안 채워졌다.그냥 계속 말하는 중이다. 아직도 안 채워졌다. 여전히 안 채워졌다.그냥 계속 말하는 중이다. 아직도 안 채워졌다. 여전히 안 채워졌다. 그냥 계속 말하는 중이다. 아직도 안 채워졌다. 여전히 안 채워졌다.그냥 계속 말하는 중이다. 아직도 안 채워졌다. 여전히 안 채워졌다.그냥 계속 말하는 중이다. 아직도 안 채워졌다. 여전히 안 채워졌다. 그냥 계속 말하는 중이다. 아직도 안 채워졌다. 여전히 안 채워졌다.그냥 계속 말하는 중이다. 아직도 안 채워졌다. 아직도 안 채워졌다. 여전히 안 채워졌다. 이제 됐나? 됐다. 여기서부턴 넘어가기 시작한다. 사진의 아래에 텍스트가 위치한 것을 확인할 수 있다.

이 경우에는 아까전에 utility class보다 좀 더 여백이 있는 모습을 확인할 수 있다. 

코드는 다음과 같다.

```html
<figure class="align-left">
  <img src="https://mmistakes.github.io/minimal-mistakes/assets/images/image-alignment-150x150.jpg" alt="">
  <figcaption>왼쪽 정렬한 모습</figcaption>
</figure>
```

---

이번엔 크기 조절을 해보자. 

<figure style="width: 300px" class="align-center">
  <img src="https://mmistakes.github.io/minimal-mistakes/assets/images/image-alignment-580x300.jpg" alt="">
  <figcaption>크기는 아까와 같이 300으로 주었다. </figcaption>
</figure> 

코드는 다음과 같다.

```html
<figure style="width: 300px" class="align-left">
  <img src="https://mmistakes.github.io/minimal-mistakes/assets/images/image-alignment-580x300.jpg" alt="">
  <figcaption>크기는 아까와 같이 300으로 주었다. </figcaption>
</figure>
```