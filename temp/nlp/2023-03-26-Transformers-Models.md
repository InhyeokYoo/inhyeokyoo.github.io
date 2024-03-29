---
title:  "Transformers: models 소개"
toc: true
toc_sticky: true
permalink: /project/nlp/transformers/models
categories:
  - NLP
  - Huggingface
  - transformers
tags:
  - Model
  - Tokenizer
  - Trainer
use_math: true
last_modified_at: 2023-03-26
---

## 들어가며


## 목표

The library was designed with two strong goals in mind:
1. Be as easy and fast to use as possible:
  - We strongly limited the number of user-facing abstractions to learn, in fact, there are almost no abstractions, just three standard classes required to use each model: configuration, models, and a preprocessing class (tokenizer for NLP, image processor for vision, feature extractor for audio, and processor for multimodal inputs).
  - All of these classes can be initialized in a simple and unified way from pretrained instances by using a common from_pretrained() method which downloads (if needed), caches and loads the related class instance and associated data (configurations’ hyperparameters, tokenizers’ vocabulary, and models’ weights) from a pretrained checkpoint provided on Hugging Face Hub or your own saved checkpoint.
  - On top of those three base classes, the library provides two APIs: pipeline() for quickly using a model for inference on a given task and Trainer to quickly train or fine-tune a PyTorch model (all TensorFlow models are compatible with Keras.fit).
  - As a consequence, this library is NOT a modular toolbox of building blocks for neural nets. If you want to extend or build upon the library, just use regular Python, PyTorch, TensorFlow, Keras modules and inherit from the base classes of the library to reuse functionalities like model loading and saving. If you’d like to learn more about our coding philosophy for models, check out our Repeat Yourself blog post.
2. Provide state-of-the-art models with performances as close as possible to the original models:
  - We provide at least one example for each architecture which reproduces a result provided by the official authors of said architecture.
  - The code is usually as close to the original code base as possible which means some PyTorch code may be not as pytorchic as it could be as a result of being converted TensorFlow code and vice versa.

  A few other goals:

    Expose the models’ internals as consistently as possible:
        We give access, using a single API, to the full hidden-states and attention weights.
        The preprocessing classes and base model APIs are standardized to easily switch between models.

    Incorporate a subjective selection of promising tools for fine-tuning and investigating these models:
        A simple and consistent way to add new tokens to the vocabulary and embeddings for fine-tuning.
        Simple ways to mask and prune Transformer heads.

    Easily switch between PyTorch, TensorFlow 2.0 and Flax, allowing training with one framework and inference with another.


## 컨셉

- Model classes can be PyTorch models (`torch.nn.Module`), Keras models (`tf.keras.Model`) or JAX/Flax models (`flax.linen.Module`) that **work with the pretrained weights** provided in the library.
- Configuration classes **store the hyperparameters required to build a model** (such as the number of layers and hidden size). You don’t always need to instantiate these yourself. In particular, if you are using a pretrained model without any modification, creating the model will automatically take care of instantiating the configuration (which is part of the model).
- Preprocessing classes convert the raw data into a format accepted by the model. A tokenizer stores the vocabulary for each model and provide methods for encoding and decoding strings in a list of token embedding indices to be fed to a model. Image processors preprocess vision inputs, feature extractors preprocess audio inputs, and a processor handles multimodal inputs.



## BertLMHeadModel

If yes then, it’s intended to be used with the EncoderDecoder model which allows you to use pre-trained encoder for as both encoder and decoder for seq2seq tasks. It’s not intended for language modeling.

While you can use that class as a standalone decoder by passing is_decoder=True to config it might not give you good results as it’s trained as an encoder.

{: .align-center}{: width="300"}
