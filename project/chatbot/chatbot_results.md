# Testing of Differnt Weights for Hybrid Search
**BM25 Keyword Search: 0.4, thenlper/gte-base Vector Search: 0.4**
## Custom Prompt Template:
```python
input_variables=['context', 'question'] template="<s> [INST] You are a helpful assistant for biomedical question-answering tasks. \n                    Use only the following retrieved context to answer the question. If the answer is not in the context or \n                    if you don't know the answer, just say that you don't know. \n                    Provide a response strictly based on the information requested in the query.[/INST] </s> \n                    [INST] Question: {question} \n                    Context: {context} \n                    Answer: [/INST]"
```

## Query:
*Is occupational outcome in bipolar disorder predicted by premorbid functioning and intelligence?*

**Execution Time:**
17.3 seconds on cuda using mistralai/Mistral-7B-Instruct-v0.1.

### Response:
 Based on the provided context, it appears that there is some debate about whether premorbid functioning and intelligence play a role in determining occupational outcome in bipolar disorder. Some studies suggest that premorbid functioning and intelligence do not have a significant impact on occupational outcome in bipolar disorder, while others suggest that they may play a role. For example, one study found that lower premorbid intelligence was associated with a greater risk of receiving disability benefits in bipolar disorder patients, but another study found that premorbid intelligence did not explain the higher risk of receiving disability benefits in bipolar disorder patients. Additionally, some studies suggest that the persistence of severe clinical symptoms may be more important in determining occupational outcome in bipolar disorder than global cognitive functioning. Overall, more research is needed to fully understand the relationship between premorbid functioning and intelligence and occupational outcome in bipolar disorder.

---

## Query:
*What are the effects of α1-antitrypsin (AAT) treatment on chronic fatigue syndrome (CFS) based on a case study involving a 49-year-old woman?*

**Execution Time:**
14.42 seconds on cuda using mistralai/Mistral-7B-Instruct-v0.1.

### Response:
 Based on the provided abstract, it appears that a 49-year-old woman with chronic fatigue syndrome (CFS) was treated with intravenous infusions of a human plasmaderived alpha1 antitrypsin (AAT) concentrate for 8 consecutive weeks. The treatment resulted in improvements in maximal workload, working memory scores, and perceptual organization scores. The patient's monocyte elastase, a regulator of inflammatory processes, also decreased to a normal range. These findings suggest a possible role for AAT in the treatment of CFS.

---

## Query:
*What are your favorite comedies?*

**Execution Time:**
8.94 seconds on cuda using mistralai/Mistral-7B-Instruct-v0.1.

### Response:
 I am sorry, but I cannot provide an answer to your question "What are your favorite comedies?" as there is no information in the given context about my personal preferences or knowledge of comedy.

---

