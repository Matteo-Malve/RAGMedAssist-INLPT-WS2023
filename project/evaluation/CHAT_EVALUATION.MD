# Evaluation of Chatmodel

## 1. Which Chatmodel to Use?

    - baseline
    - ...


## 2. Chatmodel Parameters

    - temperature
    - top-p
    - integrate memory
    - how is token limit handled
    - ...

## 3. Retriever

### 3.1 Which Retriever?

    - "normal" retriever?
    - [`MultiQueryRetriever`](https://python.langchain.com/docs/modules/data_connection/retrievers/MultiQueryRetriever)
    - ...

### 3.2 Retriever Parameters

    - top k (-> how much context?)
    - hybrid search: alpha
    - ...

# 4. Prompt Engineering

    - which prompt works best
    - answer length
    - hallucination

# 5. Eval Data

    - come up with own questions
    - find out which use case works best (causal/factual/yes-no question ... professionals/students)
    - evaluate using eval set with ground truth QAs
    - generate set using ChatGPT
    - use BLEU/ROUGE/BERTScore metrics as in assignment




