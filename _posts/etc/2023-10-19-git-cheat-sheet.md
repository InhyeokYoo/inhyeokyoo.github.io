---
title:  "실무에서 사용하는 git cheat sheet"
toc: true
toc_sticky: true
categories:
  - Git
use_math: true
last_modified_at: 2024-02-15
---

업무를 하며 유용하게 사용하는 git 명령어 정리해보았다.

## `git reset`을 활용하여 remote와 local을 일치시키기

가끔 HEAD가 detach되거나 특정 문제가 생겨 remote와 local이 일치하지 않는 경우가 발생한다.
이런 경우엔 local 저장소를 지운뒤 remote와 동기화하면 된다. 

```bash
# Remove commits to branches that doesn't exist on the remotes
git fetch --prune origin 
# reset to remote branch
git reset --hard origin/main
```

## `git add -p`와 `git commit -v`을 활용한 이력관리

### `git add -p`을 활용하여 commit을 나누자


코드를 작성하다보면 아주 당연하게도 여러 부분을 한번에 수정하게 되고, 이 중에는 commit의 단위가 다른 경우가 있을 것이다.
여태까진 `git add <file 이름>` 처럼 한 파일씩 commit을 넣게 되었는데, 한 파일의 add 단위를 분리할 수 없다는 단점이 있다.

`git add -p`는 commit의 단위를 hunk라는 작은 단위로 쪼갤 수 있을뿐만 아니라, 어느 부분이 수정되었는지 나오기에 매우 편리하다.

```bash
y - stage this hunk
n - do not stage this hunk
q - quit; do not stage this hunk or any of the remaining ones
a - stage this hunk and all later hunks in the file
d - do not stage this hunk or any of the later hunks in the file
g - select a hunk to go to
/ - search for a hunk matching the given regex
j - leave this hunk undecided, see next undecided hunk
J - leave this hunk undecided, see next hunk
k - leave this hunk undecided, see previous undecided hunk
K - leave this hunk undecided, see previous hunk
s - split the current hunk into smaller hunks
e - manually edit the current hunk
? - print help
```

개인적으로는 `s` 옵션을 통해 hunk를 잘게 나눈 뒤, `e`로 manually 작업한다.


### `git commit -v`를 활용하여 commit 변경사항을 직접 확인하기

`git commit -v`도 `git add -p`와 비슷하게 commit시 변경사항을 다시 확인하기 위해 사용한다.

해당 옵션을 사용하면 에디터에 변경사항이 포함된다.
물론 `git diff --staged`로 확인할 수도 있으나, 해당 옵션을 통해 한 번 더 변경사항을 확인할 수 있다.

Commit 메세지는 맨 윗줄에 입력하면 된다.


## 임시저장을 하고 싶을 땐 ? `git stash`!

working stage에 변경 내역이 남아있을 때 branch를 변경하는 등의 기능이 안될 때가 있다.
이럴 때는 `git stash`를 이용하여 임시저장을 하고, branch를 변경하는 등의 작업을 수행한 뒤 다시 불러오는 방식으로 작업을 이어나갈 수 있다.

```sh
# 코드 임시 저장
git stash save <stash 이름>

# 코드 임시 저장 목록 확인
git stash list

# 임시 저장 코드 가져오기
git checkout 원하는 브랜치명

# stash 내용 확인
git stash show -p <stash 이름>

# 마지막 save한 코드 가져오기
git stash apply

# stash 이름으로 apply하기
git stash apply <stash 이름>

# apply를 할 경우 list에서 사라지지 않으므로 drop을 이용하여 지워줌
git stash drop <stash 이름>

# apply와 drop을 동시에 하고 싶을 땐 pop을 사용함
git stash pop <stash 이름>
```

## Repository에서 repository를 관리하고 싶을 땐? `git submodule`!

프로젝트를 진행하다보면 여러 파드 등에서 같은 모듈이나 config 등을 사용하는 경우가 있다.
각 파드 별로 git repo를 관리하는 경우 똑같은 기능임에도 불구하고 조금씩 차이가 생길 수도 있고, 특정 repository (파드)에서만 버그를 수정하게 되는 경우 다른 repository (파드)로 이를 즉각적으로 반영하기가 어려운 경우가 많다.

이럴 때는 git submodule 기능을 사용하여 공통적인 부분을 관리하도록 하자.  
명령어는 다음과 같다.

`git submodule add <sub-module-repo.git> <추가할 디렉토리>`

이를 통해 부모 저장소 (superproject) 디렉토리 하에 자식 저장소 (submodule)을 추가할 수 있다.

### Submodule push하기

Submodule의 push는 일반적인 repo와 동일한 방식으로 진행하면 된다.

그러나 submodule에서 commit/push 이후 superproject에서 pull을 당겨야 하는 절차적 불편한 점이 있다.
이를 위해 상위 repo인 superproject에서 하위 repo(submodule)을 **직접 수정**한 후 push하는 방법도 있다.
이 경우 주의해야 할 점은,

1. Submodule의 branch가 `deatch HEAD`로 되어있는 경우가 있다. 이 경우 잘 못 push했을 때 정보가 사라질 수 있다. 따라서 반드시 branch를 변경해야 한다.
  - Branch는 해당 디렉토리에서 변경할 수도 있으며,
  - Superproject에서 `git submodule foreach git checkout master`를 통해 branch를 일괄 변경 가능하다.
2. commit 순서는 subproject -> superproject가 되야한다. 그렇지 않으면 subproject의 commit을 참조하지 않게 된다.

### Submodule pull하기

Superproject의 **디렉토리 내**에서 직접 git pull을 하거나, `git submodule update --remote`로 업데이트 할 수 있다.


### Submodule clone하기

Submodule이 포함된 repository를 온전히 clone하기 위해서는 `--recurse-submodules` 옵션을 함께 줘야한다.


## Remote branch를 가져오려면? `git switch -t`!

Git repository를 pull하면 branch는 제외하고 가져오게 된다.
만약 특정 branch를 가져오고 싶으면 `git switch -t <원격 저장소/branch 이름>`를 사용하도록 하자.


## commit을 취소하려면?

commit을 취소하기 위해선 다양한 방법을 활용할 수 있다.

### `git reset`을 활용하여 원격 저장소에 올라간 commit 취소하기

`git reset [option] <commit 이름>`을 활용하면 commit을 되돌릴 수 있다.
이후 되돌린 commit을 원격 저장소에 강제로 push하여 반영시킬 수 있다.

이 경우 **로컬 저장소의 commit 히스토리가 원격 저장소보다 뒤쳐져** 있기 때문에 push 과정에서 에러가 발생할 수 있다.
현재 작업은 뒤쳐져 있는 로컬 저장소의 commit 히스토리를 원격 저장소의 commit 히스토리로 강제로 덮어쓰는 것이므로 `-f`/`--force`을 추가하여 강제로 push할 수 있다.

이 방법은 원격 저장소에 흔적도 없이 commit들을 제거할 수 있으므로 겉보기에는 가장 깔끔한 해결책으로 보일 수 있다.
하지만 해당 branch가 팀원들과 공유하는 branch이고, commit을 되돌리기 전에 다른 팀원이 pull로 땡겨갔다면, 그때부터 다른 팀원의 로컬 저장소에는 되돌린 commit이 남아있게 된다.
이 사실을 모르는 팀원은 자신이 작업한 commit과 함께 이를 push할 것이고, 그 때 되돌렸던 commit이 다시 원격 저장소에 추가되게 될 것이다.
따라서 이 방법은 다른 팀원이 내가 되돌린 commit을 pull로 땡겨가지 않았다고 확신할 수 있는 경우에만 사용하는 것이 바람직하다.

### `git revert`을 활용하여 원격 저장소에 올라간 commit 취소하기

앞선 방법의 근본적인 문제점은 다른 팀원들과 공유하는 원격 저장소의 commit 히스토리를 강제로 조작한다는 점이다.
이를 막기위해서는 `git revert`를 사용하여 revert commit을 히스토리에 쌓는 방식으로 원격 저장소의 commit을 취소한다.

`git revert <commit>`을 사용할 경우 특정 commit의 변경 사항을 제거하는 commit을 생성하는 명령어이다.

![](https://wac-cdn.atlassian.com/dam/jcr:a6a50d78-48e3-4765-8492-9e48dec8fd2f/04%20(2).svg?cdnVersion=1267)

주의할 점은 쌓여있는 commit 히스토리대로 (stack처럼) revert해야 한다는 점이며 (즉, A -> B -> C 순서의 commit이 있으면 revert C -> revert B -> revert A), 따라서 revert한 갯수만큼 commit이 추가로 생성된다.  
단 하나의 commit만 생성하고 싶다면 `--no-commit`을 이용하여 working tree와 index (staging area)에만 적용할 수 있다.

만일 여러개의 commit을 revert하고 싶다면 `git revert --no-commit HEAD~3..`처럼 범위를 입력해도 된다.


## 과거의 특정 시점으로 돌아가고 싶을 땐? `git checkout`

과거 특정 시점의 commit으로 돌아가 테스트를 하는 등의 작업을 할 때는 다음과 같이 수행하면 된다.

```sh
# 이전 시점으로 이동
git checkout <commit>

# 다시 원래 시점으로 되돌아오기
git checkout <branch_name>
```


## 파일 단위의 변경을 진행할 때는? `reset`과 `checkout`

`git reset`에 commit을 파라미터를 넣는 것과는 다르게 파일을 인자로 넣을수도 있다.
이 경우 파일의 **stage area를 해당 커밋으로 업데이트**한다 (애초에 `reset`은 staging area를 다루는 것이다).
명령어는 `git reset [<commit>] <file`와 같다.
아래의 예시를 살펴보자.

```sh
# 각 커밋은 README.md에 1, 2, 3을 추가하는 것이다
$ git log --oneline
72b2d6b (HEAD -> main) Add 3
bbc8ca8 Add 2
0132908 Add 1
69c4f3e Create README.md

# staging area는 깔끔하다
$ git status
On branch main
nothing to commit, working tree clean

# 테스트해보자
$ git reset HEAD~1 README.md
Unstaged changes after reset:
M	README.md

# 상태를 확인해보자
$ git status
On branch main
Changes to be committed:
  (use "git restore --staged <file>..." to unstage)
	modified:   README.md

Changes not staged for commit:
  (use "git add <file>..." to update what will be committed)
  (use "git restore <file>..." to discard changes in working directory)
	modified:   README.md

# WD는 HEAD와 상태가 동일하다 (3 추가)
$ git diff README.md 
diff --git a/README.md b/README.md
index 1191247..01e79c3 100644
--- a/README.md
+++ b/README.md
@@ -1,2 +1,3 @@
 1
 2
+3

# Stage area는 commit인 `HEAD~1`과 동일한 모습(=2 추가한 상태)이다
$ git diff --cached README.md
diff --git a/README.md b/README.md
index 01e79c3..1191247 100644
--- a/README.md
+++ b/README.md
@@ -1,3 +1,2 @@
 1
 2
-3

# 말했듯이 stage area만 다루므로 커밋에는 영향이 없다
$ git log --oneline
72b2d6b (HEAD -> main) Add 3
bbc8ca8 Add 2
0132908 Add 1
69c4f3e Create README.md
```

이때 commit이 `HEAD`일 경우 생략이 가능하다.

또한, `reset`이 아닌 `checkout`을 사용하는 방법도 있다.
`checkout`을 파일 단위로 사용하면 **stage area와 working directory 모두 커밋된 내용과 일치**하도록 파일을 되돌린다.
사용법은 위와 비슷하다.

```sh
git checkout [<commit>] <file>
```

둘의 차이점은 아래의 표에 나와있다.

|                                 | Index | Workdir | WD Safe? |
|---------------------------------|-------|---------|----------|
| reset [commit] <paths>          | YES   | NO      | YES      |
| checkout [commit] <paths>       | YES   | YES     | **NO**   |

checkout은 working directory를 건들이기 때문에 주의를 기울여야 한다.


## `git reset` 후에 다시 돌아가고 싶다면?

`git reset`을 할 경우 기존의 내역이 `ORIG_HEAD`에 저장된다.
따라서 해당 커밋으로 되돌아가면 된다.

```sh
git reset --hard ORIG_HEAD
```
