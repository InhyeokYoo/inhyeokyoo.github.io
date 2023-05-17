---
title:  "Huggingface datasets 소개"
toc: true
toc_sticky: true
permalink: /project/nlp/huggingface/datasets
categories:
  - NLP
  - Huggingface
  - datasets
tags:
use_math: true
last_modified_at: 2023-04-23
---

TODO-list:
- Datasets 클래스

## `Dataset`


### `select`

temp.select(indices: Iterable, keep_in_memory: bool = False, indices_cache_file_name: Optional[str] = None, writer_batch_size: Optional[int] = 1000, new_fingerprint: Optional[str] = None) -> 'Dataset'
Docstring:
Create a new dataset with rows selected following the list/array of indices.

Args:
    indices (`range`, `list`, `iterable`, `ndarray` or `Series`):
        Range, list or 1D-array of integer indices for indexing.
        If the indices correspond to a contiguous range, the Arrow table is simply sliced.
        However passing a list of indices that are not contiguous creates indices mapping, which is much less efficient,
        but still faster than recreating an Arrow table made of the requested rows.
    keep_in_memory (`bool`, defaults to `False`):
        Keep the indices mapping in memory instead of writing it to a cache file.
    indices_cache_file_name (`str`, *optional*, defaults to `None`):
        Provide the name of a path for the cache file. It is used to store the
        indices mapping instead of the automatically generated cache file name.
    writer_batch_size (`int`, defaults to `1000`):
        Number of rows per write operation for the cache file writer.
        This value is a good trade-off between memory usage during the processing, and processing speed.
        Higher value makes the processing do fewer lookups, lower value consume less temporary memory while running `map`.
    new_fingerprint (`str`, *optional*, defaults to `None`):
        The new fingerprint of the dataset after transform.
        If `None`, the new fingerprint is computed using a hash of the previous fingerprint, and the transform arguments.

Example:

```py
>>> from datasets import load_dataset
>>> ds = load_dataset("rotten_tomatoes", split="validation")
>>> ds.select(range(4))
Dataset({
    features: ['text', 'label'],
    num_rows: 4
```


## `DatasetDict`

이름을 키값(e.g. train, test)으로하며 `Dataset` 오브젝트를 밸류로하는 딕셔너리이다.
이에는 `map`이나 `filter`와 같은 transform 메소드를 지원하며, 모든 밸류에 대해 한번에 진행할 수 있다.



{: .align-center}{: width="300"}
