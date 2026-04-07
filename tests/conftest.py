"""Shared pytest fixtures."""

import pytest


@pytest.fixture
def markdown_bytes() -> bytes:
    """A multi-section markdown document for indexing tests."""
    return b"""# Introduction

This document covers the basics of retrieval-augmented generation.

## What is RAG?

RAG combines a retrieval system with a language model.
The retrieval system fetches relevant documents from a knowledge base.
The language model generates an answer conditioned on those documents.

### Key Components

The three key components are: retriever, reader, and knowledge base.

## Why Use RAG?

RAG reduces hallucinations by grounding answers in retrieved facts.
It also allows the knowledge base to be updated without retraining the model.

# Architecture

The architecture consists of two main phases: indexing and querying.

## Indexing Phase

Documents are split into chunks and embedded into a vector store.

## Querying Phase

At query time the user question is embedded and compared against chunk vectors.
Top-K chunks are retrieved and passed to the LLM as context.
"""


@pytest.fixture
def flat_bytes() -> bytes:
    """A flat document with no markdown headings (triggers fallback parser)."""
    return (
        b"Retrieval-augmented generation (RAG) is a technique that combines "
        b"neural retrieval with conditional text generation. "
        b"It was introduced to improve factual accuracy of language models. "
        b"The retriever fetches relevant passages from a corpus. "
        b"The generator produces a response conditioned on those passages."
    )
