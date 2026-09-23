---
title: "KV Cache + GPUs + thoughts"
description: "Going from KV cache intuition to bigger-picture inference questions."
date: "2026-09-23"
draft: false
tags:
  - ml
  - inference
  - journey
---

After working through transformer basics, I wanted to better understand what matters during inference in practice, not just in theory.

## What I explored

I spent time coding a transformer in PyTorch from scratch to solidify my understanding, and I learned how KV cache, prefill, and decode work together. Similar to the PyTorch-attention Jupyter notebook from my previous blog, I also started one for KV cache, but I did not finish it yet. Even so, it was helpful to have that hands-on side instead of only reading, because it made the inference flow feel much more concrete.

However, once I got comfortable with the basic concept, it did not seem too hard conceptually, and that pushed me toward a bigger question: how do these things actually run on GPUs? That led me to Horace He's [How to Scale Your Model](https://horace.io/brrr_intro.html), which was a great read and introduced ideas like memory-bound vs. compute-bound workloads. I supplemented that with some of my own reading on how GPUs execute these inference workloads.

At the same time, I found a lot of great resources, but I also ran into the problem of too much choice. I started going through the [JAX Scaling Book (Inference)](https://jax-ml.github.io/scaling-book/inference/), [Tiny LLM](https://skyzh.github.io/tiny-llm/), and [Transformer Inference Arithmetic](https://kipply.github.io/blog/transformer-inference-arithmetic/). I kept pausing because I would wonder whether another resource might be a more efficient path for learning. As soon as I got a little deeper, I kept questioning whether it was really the most optimal thing for me to be doing.

I want to get to the interesting parts and build a cool project. It is still early, but naturally I am a bit impatient. So for the coming days, I want to be more structured because my studying feels scattered right now. I will commit to one project and use outside articles as support instead of constantly switching.

I am going to put out a blog post benchmarking a naive Hugging Face `transformers` server vs. the vLLM inference engine. The goal is to measure how much better vLLM is and understand why.



## Useful links

- [How to Scale Your Model](https://horace.io/brrr_intro.html)
- [JAX Scaling Book: Inference](https://jax-ml.github.io/scaling-book/inference/)
- [Tiny LLM](https://skyzh.github.io/tiny-llm/)
- [Transformer Inference Arithmetic](https://kipply.github.io/blog/transformer-inference-arithmetic/)

## Download (optional)

[Download KV cache notebook (.zip)](/pytorch-kv-cache.zip)
