# 👾 RAGMedAssist

### 🧑🏻‍🎓 Team Members

| Name and surname    | Matric. Nr. | Course of study                            |   e-mail address   |
|:--------------------|:------------|:-------------------------------------------|:-------------------|
| Matteo Malvestiti | 4731243     | M.Sc. Data and Computer science (Erasmus) | matteo.malvestiti@stud.uni-heidelberg.de|
| Sandra Friebolin | 3175035     | B.Sc. Computational linguistics | sandra_friebolin@proton.me |
| Yusuf Berber | 4736316     | M.Sc. Data and Computer Science <span style="color:red"> **(?)** </span> | yusuf.berber@stud.uni-heidelberg.de |


### Advisor

Robin Khanna (R.Khanna@stud.uni-heidelberg.de)

***

## Table of contents

1. ➡️ [Introduction](#introduction)
2. 📚 [Related Work](#related-work)
3. ⚙️ [Approach](#approach)
    - ✂️ 3.1 [Data Processing](#data-processing)
    - 👾 3.2 [Algorithms & Methods](#algorithms-methods)
    - 3.3 [Baselines](#baselines)
    - 3.4 [Fine-Tuning](#fine-tuning)
4. 🔬 [Experimental Setup & Results](#experimental-setup-results)
    - 💽 4.1 [Data](#data)
    - 📥 4.2 [Vector Database](#vector-database)
    - 📈 4.3 [Evaluation](#evaluation)
        - a) [Evaluation of Information Retrieval](#retrieval-eval)
        - b) [Evaluation of Chatmodel](#chatmodel-eval)
    - 🧐 4.4 [Analysis](#analysis)
5. ⚡️ [Limitations & Future Work](#limitations-future-work) 🔮
6. 💡[Conclusion](#conclusion)
7. 💻 [References](#references)
8. 📊 [Appendix](#appendix)
    - 🧑🏻‍🎓 [Contributions](#contributions)
    - 📝 [Anti-Plagiarism Declaration](#anti-plagiarism)

***

# <a name="introduction"></a>1. ➡️ Introduction

Navigating the complexities of medical information, especially when it is laden with technical jargon, can be overwhelming yet essential for making critical health decisions. Our system bridges this gap by simplifying the intricate world of medical knowledge. It allows users to ask questions in everyday language and provides informed, understandable answers derived from a comprehensive medical dataset.

By also citing sources, our system not only educates but empowers users to verify and trust the information, facilitating more informed health decisions. The target audience for our project is [...]

Our focus is on leveraging generative AI with Retrieval Augmented Generation (RAG) techniques to efficiently navigate through 60,000 PubMed article abstracts on intelligence. This approach overcomes the limitations of traditional keyword searches by using a hybrid search algorithm. It combines semantic retrieval, using dense vector search for relevance based on cosine similarity, with keyword search for domain-specific terms. Our innovative use of Pinecone's hybrid search integrates a sparse-dense index, optimizing accuracy and preventing misinformation. 

- outline of our approach 
- outlook on results

<!-- maybe it is best written at the end, since we don't know exactly what our results will be/which exact problem (students/professionals..) we address :D 

I would not put the following in the introduction. I think it should only briefly outline our approach without technical details :) 

"""as it will be better explained in sec.3 [Approach]. The code is entirely written in python and we make use of the incredible langchain lybrary.
Before arriving at this point though, a huge amoutn of work was spent on the retrieval of the dataset, on the search of a good database and on the choice of the embedding model. The latter was supported by a big evaluation process, distinguished in two phases: quantitative and qualitative. More detailes are recorded in section 3 and in the README of the evaluation folder.
[...]"""
-->

# <a name="related-work"></a>2. 📚 Related Work
<!--
- put our work into context of current research
- including papers read for research/that used same techniques but applied to different problems
- emphasize how our work differs from previous work, outlining their limitations/why our application domain is different
- ⚠️ only major points, not too much detail
-->
- Leaderboard paper on embedding models
- sth about Perplexity.ai ? (similar system to ours)
- A feature that we implemented and we will discuss in [sec. 3.2](#iii-document-retrieval) is Ensemble Retrieval. More specifically, we retrieved documents both with a sparse retriever and a dense retriver and reranked them with Reciprocal Rank Fusion. Inspiration and guideline for this procedure was [Cormack et al., 2009](#RRF)

# <a name="approach"></a>3. ⚙️ Approach
<!--
- conceptual details of our system (about its functionality, its components, data processing pipelines, algorithms, key methods)
- 💡 be specific about methods (include equations, show figures...)
- 💡 emphasize creative/novel parts, but also properly cite existing methods
- 💡 stick to fixed vocabulary (mathematical notations, method & dataset names) and writing style!
- describe baseline approaches (briefly if from external source)
-->

## <a name="data-processing"></a>3.1 ✂️ Data Processing

Several data cleaning and pre-processing strategies were considered and applied according to their usefulness to our specific application (see [`preprocess_data.py`](data/preprocess_data.ipynb)):

✅ **Removing Special Characters:** This includes stripping out unnecessary punctuation, symbols, or special characters that are not relevant to the analysis or could interfere with the model's understanding of the text. We apply this step to to enhance data consistency and reduce noise, thereby improving model focus and efficiency.

✅ **Normalization:** This process standardizes text to a consistent format, enhancing data uniformity and simplifying processing. Specifically, we implemented Unicode normalization to transform author names with special characters — common in languages such as Swedish and French — into a uniform representation. 

✅ **Tokenization:** Essential to the embedding process, tokenization divides text into manageable pieces, or tokens. Our chosen embedding models employ unique tokenizers that segment text into words and subwords. This granularity enables precise interpretation of biomedical terminology, accommodating the field's extensive vocabulary and specialized jargon.

✅ **Removing Short Abstracts:** Abstracts with fewer than 100 characters often lack sufficient detail, offering minimal insight. To enhance our dataset's quality, we excluded such brief abstracts and those lacking any abstract text. This refinement process resulted in the retention of 58,535 abstracts, effectively removing 319 from our initial collection.

❌ **Lowercasing:** Converting all text to lowercase can be beneficial for consistency and improve performance for many embedding models. However, in certain scientific contexts, such as our biomedical texts, case sensitivity is important, for instance when distinguishing between gene names and common words. We thus decided against lowercasing, also given that we used very advanced embedding models able to handle case sensitivity.

❌ **Handling Stop Words:** Removing common words that do not contribute much meaning to the sentences was a common practice for traditional approaches as it gives more focus on important content words. However, advanced embedding models, especially those based on Transformer architecture, are designed to capture the context of the entire sentence or document, including the function of stop words. In fact, [Miyajiwala et al.](#stopwords) (2022) have shown that the removal can even lead to a drop in performance. We thus decided against removing them.

❌ **Handling Bigrams or N-grams:** Advanced Transformer based models do not require this step since they are designed to capture word context using their attention mechanisms and positional embeddings, making explicit n-gram creation less necessary.

## <a name="algorithms-methods"></a>3.2 👾 Algorithms & Methods
<!--
- Overall Approach
-->
Our ultimate goal is to create a chatbot capable of answering questions on topics within the biomedical domain and a specific focus on intelligence-related matters. To achieve this, we have developed a Retrieval-Augmented-Generation (RAG) system that consists of two main components: a retrieval system and a generative large language model (LLM). The retrieval system is responsible for retrieving the most relevant documents for a given user query. Its effectiveness and accuracy is guaranteed thanks to our adoption of the [FAISS](https://ai.meta.com/tools/faiss/) vector database for storing our document embeddings. A central part of our chatbot's ability to generate contextually relevant questions is the chosen LLM, which leverages context provided by the retrieved documents to produce not only accurate responses but also tailored to the specific nuances of the users' query. The versatile [LangChain](https://python.langchain.com/docs/get_started/introduction) framework we use facilitates the seamless integration of these two components. It allows us to dynamically integrate context and reasoning capabilities into our application, thereby enabling the development of an advanced, interactive AI system.

In the subsequent sections, we will delve into the specifics of the algorithms and techniques we have employed within our system.

#### I. Embedding Documents

The initial phase of developing our system entailed selecting an appropriate embedding model to generate document embeddings, a critical component for ensuring the accuracy of document retrieval and, by extension, the quality of the responses generated by our system. To this end, we referred to the [Massive Text Embedding Benchmark (MTEB) Leaderboard](https://huggingface.co/spaces/mteb/leaderboard) for retrieval tasks. Our selection criteria prioritized models that were both high-performing according to the leaderboard and feasible in terms of availability and size. Please note that this leaderboard is continuously updated with new models, so our chosen one might not keep its top place currently. After a thorough examination of various models, detailed in section 4.3 a) [Evaluation of Information Retrieval](#retrieval-eval), we chose the [General Text Embedding (GTE)](https://huggingface.co/thenlper/gte-large) model, which easily accessible via Hugging Face's platform:

```py
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("thenlper/gte-large")
```

The GTE model, developed by Alibaba DAMO Academy, is based on the BERT architecture and optimized for generating high-quality embeddings. It has a compact size of 0.67 GB, consists of 335 million parameters, and supports embedding vectors of 1024 dimensions with a maximum token input length of 512. This model's training involved a large collection of relevant text pairs across various domains, making it particularly adept at capturing semantic textual similarities. Such capabilities make it exceptionally suitable for our application, where accurate semantic understanding is crucial.

Given the nature of our work with abstract texts, we often encounter documents that exceed the 512 maximum token length capacity of our chosen embedding model. To address this challenge, we employed LangChain's [Text Splitters](https://python.langchain.com/docs/modules/data_connection/document_transformers/) functionality to divide our documents into smaller chunks that the embedding model can manage without exceeding its token limit. Specifically, our approach involves a recursive and token-count-based splitting process, utilizing LangChain's integration with `tiktoken` for its efficient Byte Pair Encoding (BPE) tokenizer capabilities. This tool provides an accurate estimation of the tokens present in the text, ensuring that each document segment conforms to the model's constraints. Additionally, we leverage LangChain's `RecursiveCharacterTextSplitter`, which guarantees that no split exceeds the permissible chunk size in tokens. This splitter ensures that if any segment is too large, it will be further divided recursively until all parts are within the acceptable size range. To maintain coherence and context continuity across the document segments, we incorporated an overlap of 100 tokens between each chunk. 

```js
text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(chunk_size=400,
                                                                    chunk_overlap=100,)
chunked_docs = text_splitter.split_documents(docs_all)
```

This way we transformed our collection of 58,535 original abstracts into 62,615 chunked documents that are embedded using the GTE model and uploaded to be stored in our vector database.

#### II. Vector Storage

#### III. Document Retrieval

We integrated Langchain's `EnsembleRetriever` into our search framework to make use of a hybrid model that combines BM25-based keyword search with vector search to provide precise and contextually relevant results. This approach is particularly beneficial for datasets dealing with highly specific terms, such as our biomedical abstracts, where keyword search excels in precision. By leveraging the strengths of both methodologies, we ensure users receive accurate information that not only aligns with their query's intent but also navigates the complexities of specialized terminology. 

Keeping the context token limit in mind, we pass to the chain only a reasonable number of abstracts: we use the ensemble retriever, we rank them with RRF and finally we only keep the first $topk_{RRF}$, a parameter defaulted to three, but that can be specified in the [configuration file](chatbot/cfg.yaml).

#### IV. Chatmodel Configuration & Integration

#### V. Innovative Aspects & Technical Choices

## <a name="baselines"></a>3.3 Baselines


## <a name="fine-tuning"></a>3.4 Fine-Tuning

After developing and evaluating the embedding models for our retrieval system, we initially opted against finetuning. Our chosen embedding model, `thenlper_gte-base`, showed high performance, with metrics above 95% in preliminary evaluations. However, upon advisor recommendation, we explored finetuning and investigated two different methods for unsupervised learning. First, we applied the Transformer-based Sequential Denoising Auto-Encoder (TSDAE) method that is centered around the idea to construct an original sentence from its corrupted one (see [`TSDAE.py`](finetuning/TSDAE.py)). During training, corrupted sentences are encoded into fixed-sized vectors and reconstructed by the decoder into the original sentence ([Wang et al., 2021](#TSDAE)). As a second method we explored contrastive learning in the context of finetuning and created positive and negative training samples for this purpose (see [`create_contrastive_learning_data.py`](finetuning/create_contrastive_learning_data.py)). For the positive one we used the paraphrasing model [`tuner007/pegasus_paraphrase`](https://huggingface.co/tuner007/pegasus_paraphrase) which is finetuned for paraphrasing tasks. The idea behind this approach is to teach the model to differentiate between paraphrased (positive) and unrelated (negative) sentence pairs. 

Upon further consultation with our advisor though, we decided not to keep this finetuning data for future work, but did not carry out any further experiments, given also the danger of increasing hallucinations in the model's output after finetuning.

# <a name="experimental-setup-results"></a>4. 🔬 Experimental Setup & Results

## 4.1 💽 Data

Our chosen dataset comprises abstracts and associated metadata from medical articles sourced from [PubMed](https://pubmed.ncbi.nlm.nih.gov/?term=intelligence+%5BTitle%2Fabstract%5D&filter=simsearch1.fha&filter=years.2013-2023&sort=date), a free search engine for life sciences and biomedical literature, managed by the U.S. National Library of Medicine at the National Institutes of Health. To manage time and computational constraints, our focus is limited to abstracts published between 2013 and 2023 featuring the keyword "intelligence", totaling 58,854 documents.

The documents in the dataset follow a structured format typical of biomedical literature. Each document contains several key elements that have designated abbreviations:

- **PMID (PubMed Identifier):** A unique number assigned to each PubMed record, used for easy reference and retrieval.
- **Title:** The title of the article.
- **Abstract:** A brief summary of the research, methods, results, and conclusions. It's a crucial part of the document as it provides the essence of the research without the need to read the full article.
- **Authors:** Lists the full names and initials of the authors, along with their affiliations, providing information about who conducted the research and their institutional backgrounds.
- **Date of Publication:** Indicates when the article was published, which is important for understanding the timeliness and relevance of the research.
- **DOI (Digital Object Identifier):** A unique alphanumeric string assigned to the document, providing a permanent link to its location on the internet.
- **Additional Information:** Includes various bibliographic details like journal name (JT), issue (IP), volume (VI), language of the article (LA), grant and funding information (GR), and publication type (PT). However, we do not plan to use these as metadata for our application.

The metadata selected for our project encompasses the authors, title, date, and DOI of each document, as illustrated in this data point example:

<p align="left">
  <img src="./docs/images/datapoint_example.png" width="700" />
</p>

The abstracts, serving as the core of our dataset, will be utilized by our retrieval system to identify and present the most pertinent documents in response to user queries, thereby forming the basis for generating informed and accurate answers by our chosen LLM. Metadata such as the DOI not only aids in establishing the credibility and context of the research but also enables our system to link directly to the source in the answers it generates - an additional functionality of our system.

We acquired the data on January 4, 2024, via the `BioPython Entrez` API, which is a tool for fetching scholarly data efficiently, using the following query:

```py
query = f"intelligence[Title/Abstract] AND (\"{year}/{month_start}\"[Date -  Publication] : \"{year}/{month_end}\"[Date - Publication])"
```

We downloaded the data in XML format and segmented the retrieval quarterly across different years to sequentially gather the required dataset in manageable batches, ensuring comprehensive data collection without overstepping the API's limitations. See [`download_pubmed_data.ipynb`](data/download_pubmed_data.ipynb) for details.

Following the data preprocessing steps, we conducted an in-depth analysis to extract meaningful insights about our dataset (see [`data_analytics.ipynb`](data/data_anaylitics.ipynb)). We discovered that abstract lengths show a wide range, with the shortest being 93 characters and the longest reaching 60,664 characters. The average abstract length stands at 1,504.78 characters. The histogram presented below illustrates the distribution of abstract lengths, highlighting how frequently each length occurs.

<p align="left">
  <img src="./docs/images/distribution_abstract_length_log_scale.png" width="700" />
</p>

Turning our attention to the publication frequency on the topic of "intelligence," we noted a growing interest over time. The barplot below visualizes this upward trend, clearly showing a year-over-year increase in the number of publications, which signals a growing engagement with the topic.

<p align="left">
  <img src="./docs/images/distribution_publications_over_time.png" width="700" />
</p>

The visual analysis of publications per author reveals a common trend: the majority of authors contribute fewer than 2 publications on average, highlighting a broad base of singular contributions within the field. Notably, the most prolific author has made an impressive 94 contributions.

<p align="left">
  <img src="./docs/images/distribution_authors_frequency_contribution.png" width="700" />
</p>

[Ian J Deary](https://www.research.ed.ac.uk/en/persons/ian-deary-2) stands out as the leading author, reflecting his extensive involvement in intelligence and cognitive aging research. The following visualization ranks the top 10 authors by their number of publications. Notably, 'Unknown' appears as the fourth-highest entry, signaling some unidentified authors within the dataset.

<p align="left">
  <img src="./docs/images/distribution_authors_most_frequent.png" width="700" />
</p>

For analyzing common themes appearing in our dataset based on the titles of publications, Latent Dirichlet Allocation (LDA) was used as a simple method for extracting latent topics ([Blei et al., 2003](#LDA)). We identified two topics, the first of which displays prominent terms such as "study", "ai" and "chatgpt", pointing to a strong emphasis on artificial intelligence research. The second topic focuses on terms like "cognitive", "effect", "brain", "patient" and "disorder", indicating a research concentration on cognitive associations, possibly in developmental or clinical contexts. Interestingly, despite the obvious prevalence of the term "intelligence" across our documents, the prominent emergence of artificial intelligence as a distinct theme was a notable discovery.

<p align="left">
  <img src="./docs/images/topics_LDA.png" width="700"/>
</p>


Your document is well-structured and provides a clear comparison between Pinecone and FAISS as vector databases. Below are some proofreading suggestions to enhance clarity, grammar, and consistency:

## <a name="vectorstore"></a>4.2 📥 Vector Database


We compared two vector databasesm FAISS (local) and Pinecone (cloud-based), for our project. To do this, we created two retrievers with the same configurations: one uses FAISS and the other Pinecone as the vector store. The evaluation was conducted over a set of 167 queries, comparing the performance based on the following metrics:

Firstly, we compared the speed of both retrievers. It turned out that FAISS retrieves the `topk` context for all 167 instances in only 4 seconds, while Pinecone takes over 40 seconds.

<p align="left">
  <img src="./evaluation/compare_retrievers/images/total_execution_time_plot.png" width="400"/>
</p>

The comparison occurred with a stable internet connection. Because FAISS is a local vector store, it is significantly faster than Pinecone.

For each query, we also had the correct context, which was generated based on that context. Secondly, we compared the percentage of times the correct context was among the retrieved documents for different `topk` values. As expected, the result was almost identical for both vector stores since they use the same embeddings.

<p align="left">
  <img src="./evaluation/compare_retrievers/images/success_percentage_plot.png" width="400"/>
</p>

Below, you can view the results presented in tables:

**Results for FAISS:**

| Metric | Execution Time per Query | Total Execution Time | Match Count | Success Percentage |
|--------|--------|-------|--------|-------|
| k=1 | 0.0236 seconds | 3.9481 | 152 | 91.02% |
| k=2 | 0.0231 seconds | 3.8656 | 158 | 94.61% |
| k=3 | 0.0246 seconds | 4.1032 | 163 | 97.60% |
| k=4 | 0.0220 seconds | 3.6683 | 163 | 97.60% |
| k=5 | 0.0212 seconds | 3.5427 | 164 | 98.20% |

**Results for Pinecone:**

| Metric | Execution Time per Query | Total Execution Time | Match Count | Success Percentage |
|--------|--------|-------|--------|-------|
| k=1 | 0.2125 seconds | 35.4944 | 152 | 91.02% |
| k=2 | 0.2003 seconds | 33.4454 | 159 | 95.21% |
| k=3 | 0.2309 seconds | 38.5603 | 162 | 97.01% |
| k=4 | 0.2165 seconds | 36.1497 | 163 | 97.60% |
| k=5 | 0.2098 seconds | 35.0408 | 164 | 98.20% |

In summary, FAISS met all our requirements and proved to be faster than Pinecone. FAISS can retrieve relevant documents in just 0.02 seconds. The only disadvantage was that we need to store our FAISS indices locally, which corresponds to almost 200MB. Pinecone is a commercial vector store and will be actively developed. It offers more functions than FAISS, such as ensemble retriever or metadata filtering, but these extra functions can only be accessed with a paid account.


## <a name="evaluation"></a>4.3 📈 Evaluation 
<!-- 
- explain & define used/own metrics 
- motivate expected achievements
-->

### <a name="retrieval-eval"></a>a) Evaluation of Information Retrieval

For the quantitative and qualitative evaluation of our retrieval system, we made use of the [PubMedQA](https://pubmedqa.github.io). This dataset contains [1,000 expert-labeled questions](https://github.com/pubmedqa/pubmedqa/blob/master/data/ori_pqal.json) together with both long and short answers, such as "yes/no", as well as the context and PMID. Unfortunately, only 176 instances from our "Intelligence 2013-2023" dataset we use for context retrieval are contained in this evaluation dataset. We use these instances for our experiments.

#### I. Quantitative Evaluation

We compute accuracy (see [`evaluate_embeddings_accuracy.ipynb`](evaluation/retrieval_evaluation/quantitative_evaluation/evaluate_embeddings_accuracy.ipynb)), F1 score, mean reciprocal rank (MRR), and normalized discounted cumulative gain (nDCG) (see [`compute_mrr_ndcg_f1.ipynb`](evaluation/retrieval_evaluation/quantitative_evaluation/compute_mrr_ndcg_f1.ipynb)). For these quantitative experiments, we compare the PMID of our retrieved documents with the ground truth PMID. The evaluated embedding models were chosen from the [HuggingFace Leaderboard for Retrieval](https://huggingface.co/spaces/mteb/leaderboard) based on their performance but also their size (some advanced models were too large for our resources). We used [Faiss](https://engineering.fb.com/2017/03/29/data-infrastructure/faiss-a-library-for-efficient-similarity-search/) as our vector database for the experiments since it is deterministic and thus makes comparable results possible.

**Accuracy:** We considered different values of top `k` retrieved results. Since we retrieve three documents as context for our chat model, we focus the analysis on `k=3`. The best performing models under this configuration are `thenlper_gte-base`, `BAAI_bge-base-en-v1.5` and `jamesgpt1_sf_model_e5`. Keyword search via `BM25` was deployed as a baseline to compare against our semantic search methods (see [`compare_against_keyword_search.ipynb`](evaluation/retrieval_evaluation/quantitative_evaluation/compare_against_keyword_search.ipynb)).

|    **Accuracy**                              |   k=1 |   k=2 |   **k=3** |   k=4 |   k=5 |   k=6 |   k=7 |   k=8 |   k=9 |   k=10 |   k=11 |   k=12 |   k=13 |   k=14 |   k=15 |   k=16 |   k=17 |   k=18 |   k=19 |   k=20 |
|:---------------------------------|------:|------:|------:|------:|------:|------:|------:|------:|------:|-------:|-------:|-------:|-------:|-------:|-------:|-------:|-------:|-------:|-------:|-------:|
| `BM25`              | 0.635  | 0.707 | 0.731  | 0.754 | 0.772 | 0.802 | 0.808 | 0.826  | 0.832  |  0.844  |  0.85  |  0.856 |  0.862 |  0.862 |  0.862 |  0.862 |  0.862 |  0.862 |  0.868 |  0.868 |
| `dmis-lab_biobert-base-cased-v1.1` | 0.084 | 0.114 | 0.168 | 0.192 | 0.198 | 0.204 | 0.216 | 0.24  | 0.251 |  0.257 |  0.275 |  0.287 |  0.287 |  0.287 |  0.293 |  0.299 |  0.299 |  0.299 |  0.317 |  0.323 |
| `all-MiniLM-L6-v2`                 | 0.683 | 0.838 | 0.856 | 0.88  | 0.898 | 0.928 | 0.934 | 0.94  | 0.94  |  0.946 |  0.952 |  0.958 |  0.964 |  0.964 |  0.97  |  0.976 |  0.976 |  0.976 |  0.976 |  0.976 |
| **`BAAI_bge-base-en-v1.5`**            | 0.85  | 0.94  | **0.964** | 0.976 | 0.982 | 0.982 | 0.982 | 0.982 | 0.982 |  0.982 |  0.982 |  0.982 |  0.982 |  0.988 |  0.988 |  0.988 |  0.988 |  0.988 |  0.988 |  0.988 |
| `llmrails_ember-v1`                | 0.85  | 0.934 | 0.958 | 0.964 | 0.97  | 0.976 | 0.982 | 0.982 | 0.982 |  0.982 |  0.982 |  0.982 |  0.982 |  0.982 |  0.982 |  0.982 |  0.982 |  0.982 |  0.982 |  0.982 |
| **`jamesgpt1_sf_model_e5`**            | 0.856 | 0.922 | **0.964** | 0.97  | 0.97  | 0.982 | 0.982 | 0.982 | 0.982 |  0.982 |  0.982 |  0.982 |  0.982 |  0.982 |  0.982 |  0.982 |  0.982 |  0.982 |  0.982 |  0.982 |
| **`thenlper_gte-base`**                | 0.91  | 0.946 | **0.976** | 0.976 | 0.982 | 0.994 | 0.994 | 0.994 | 0.994 |  0.994 |  0.994 |  0.994 |  0.994 |  0.994 |  0.994 |  0.994 |  0.994 |  0.994 |  0.994 |  0.994 |
| `intfloat_e5-base-v2`              | 0.79  | 0.904 | 0.94  | 0.958 | 0.958 | 0.964 | 0.964 | 0.97  | 0.97  |  0.97  |  0.97  |  0.976 |  0.982 |  0.982 |  0.982 |  0.982 |  0.982 |  0.982 |  0.982 |  0.988 |

The following plots are arranged in descending order based on the performance of the models, displaying the three best-performing models first.

<p float="left">
  <img src="evaluation/retrieval_evaluation/quantitative_evaluation/images/retriever_accuracy_thenlper_gte-base.png" width="300" />
  <img src="evaluation/retrieval_evaluation/quantitative_evaluation/images/retriever_accuracy_BAAI_bge-base-en-v1.5.png" width="300" /> 
  <img src="evaluation/retrieval_evaluation/quantitative_evaluation/images/retriever_accuracy_jamesgpt1_sf_model_e5.png" width="300" />
</p>

<p float="left">
  <img src="evaluation/retrieval_evaluation/quantitative_evaluation/images/retriever_accuracy_llmrails_ember-v1.png" width="300" /> 
  <img src="evaluation/retrieval_evaluation/quantitative_evaluation/images/retriever_accuracy_e5-base-v2.png" width="300" />
  <img src="evaluation/retrieval_evaluation/quantitative_evaluation/images/retriever_accuracy_all-MiniLM-L6-v2.png" width="300" /> 
</p>


<p float="left">
  <img src="evaluation/retrieval_evaluation/quantitative_evaluation/images/retriever_accuracy_all-MiniLM-L6-v2.png" width="300" />
  <img src="evaluation/retrieval_evaluation/quantitative_evaluation/images/keyword_search_bm25_accuracies.png" width="300" />
  <img src="evaluation/retrieval_evaluation/quantitative_evaluation/images/retriever_accuracy_biobert.png" width="300" />
</p>

**F1 Score:** We decided to only further evaluate the top 5 models plus the `BM25` baseline, using the F1 score to investigate each model's balance between relevance (precision) and completeness (recall). The F1 score ranges from 0 to 1, where 1 indicates perfect precision and recall. This measure highlights the importance of the quality of top-ranked documents, as the F1 score tends to decrease when more results are included, underlining the significance of high-quality initial results in retrieval systems. `thenlper_gte-base` again stands out as the top performing model, consistently maintaining the highest scores.

|     **F1 Score**                  |   k=1 |   k=2 |   **k=3** |   k=4 |   k=5 |   k=6 |   k=7 |   k=8 |   k=9 |   k=10 |
|:----------------------|------:|------:|------:|------:|------:|------:|------:|------:|------:|-------:|
| `BM25`   | 0.635  | 0.471 | 0.365 | 0.302 | 0.257 | 0.229 | 0.202 | 0.184 | 0.166 |  0.154 |
| `BAAI_bge-base-en-v1.5` | 0.85  | 0.643 | 0.5   | 0.4   | 0.333 | 0.286 | 0.25  | 0.222 | 0.2   |  0.182 |
| **`llmrails_ember-v1`**     | 0.85  | 0.639 | **0.503** | 0.402 | 0.335 | 0.287 | 0.251 | 0.224 | 0.201 |  0.183 |
| **`jamesgpt1_sf_model_e5`** | 0.856 | 0.635 | **0.506** | 0.407 | 0.339 | 0.294 | 0.257 | 0.229 | 0.206 |  0.187 |
| **`thenlper_gte-base`**     | 0.91  | 0.651 | **0.512** | 0.41  | 0.345 | 0.299 | 0.262 | 0.233 | 0.21  |  0.191 |
| `intfloat_e5-base-v2`   | 0.79  | 0.619 | 0.485 | 0.388 | 0.323 | 0.277 | 0.243 | 0.216 | 0.194 |  0.176 |

The following plots are again arranged in descending order, based on the performance of the models, displaying the three best-performing models first.

<p float="left">
  <img src="evaluation/retrieval_evaluation/quantitative_evaluation/images/retriever_f1_thenlper_gte-base.png" width="300" />
  <img src="evaluation/retrieval_evaluation/quantitative_evaluation/images/retriever_f1_jamesgpt1_sf_model_e5.png" width="300" />
  <img src="evaluation/retrieval_evaluation/quantitative_evaluation/images/retriever_f1_llmrails_ember-v1.png" width="300" /> 
</p>
  
<p float="left">
  <img src="evaluation/retrieval_evaluation/quantitative_evaluation/images/retriever_f1_BAAI_bge-base-en-v1.5.png" width="300" />
  <img src="evaluation/retrieval_evaluation/quantitative_evaluation/images/retriever_f1_intfloat_e5-base-v2.png" width="300" /> 
  <img src="evaluation/retrieval_evaluation/quantitative_evaluation/images/keyword_search_bm25_F1.png" width="300" /> 
</p>

**MRR:** For MRR, the consideration of varying k is not applicable because the metric is singularly focused on how well a system ranks the first piece of relevant information - whether that relevant item appears at rank 1 or any other position. MRR captures this by measuring the average inverse rank of the first relevant document across all queries. Our results show that on average, the first relevant or correct answer tends to be very close to the top position in the search results, with `thenlper_gte-base` repeatedly displaying top performance, emphasizing its capability in not just identifying relevant documents but also in ranking the most relevant document as close to the top position as possible, which is important for retrieval systems. The baseline in contrast has a much lower score.

|    **MRR**                   |     |
|:----------------------|------:|
| `BM25`   | 0.7 |
| **`BAAI_bge-base-en-v1.5`** | **0.906** |
| `llmrails_ember-v1`     | 0.9   |
| **`jamesgpt1_sf_model_e5`** | **0.905** |
| **`thenlper_gte-base`**   | **0.938** |
| `intfloat_e5-base-v2`   | 0.864 |

**nDCG:** Our nDCG evaluation was limited to `k=1,2,3` in order to mirror the operational constraints of our later chat model, which only retrieves the top three documents. It provides insight into how well our retrieval system ranks relevant documents at the top of its search results. A gradual increase in nDCG scores from `k=1` to `k=3`for all models illustrates that while the very first document might not always be the most relevant, the system generally ranks highly relevant documents within the top three positions. `thenlper_gte-base` again slightly outperforms the other models across all `k`.

|   **nDCG**                    |   k=1 |   k=2 |   k=3 |
|:----------------------|------:|------:|------:|
| `BM25` | 0.4 | 0.436 | 0.447 |
| `BAAI_bge-base-e n-v1.5` | 0.536 | 0.593 | 0.609 |
| **`llmrails_ember-v1`**     | 0.536 | 0.59  | **0.611** |
| **`jamesgpt1_sf_model_e5`**     | 0.54 | 0.588  | **0.614** |
| **`thenlper_gte-base`**     | 0.574 | 0.607  | **0.628** |
| `intfloat_e5-base-v2`      | 0.499 | 0.567  | 0.586 |

#### II. Qualitative Evaluation

We selected a set of 10 queries from our QA dataset for further qualitative evaluation. Specifically, we chose the eight semantically most dissimilar queries (via cosine distance) to cover a broad range, and additionally the shortest and longest ones as edge cases (see [`qualitative_evaluation.ipynb`](project/evaluation/retrieval_evaluation/qualitative_evaluation/qualitative_evaluation.ipynb)). Using FAISS to embed queries and retrieve the top three most similar documents. For every model and query we investigate if the correct abstract was among the retrieved results by comparing the PMIDs.

Detailed results can be found here: [`correct_retrieval.jpg`](project/evaluation/retrieval_evaluation/qualitative_evaluation/images/correct_retrieval.jpg). As can be seen in the table below, the correct document was among the three top results for all models except `dmis-lab_biobert-base-cased-v1.1` and `all-MiniLM-L6-v2`. We consequently discarded them from further experiments due to their inaccuracy.

<img src="evaluation/retrieval_evaluation/qualitative_evaluation/images/correct_retrieval.png" width="700" />

ℹ️ Please note that we could not execute the `Muennighoff/SBERT-base-nli-v2` (SGPT) model as previously planned due to its large size. 

💡 Taking into account the results of our quantitative evaluation, we decided to proceed with the qualitative evaluation only of the three top-performing models: `BAAI/bge-base-en-v1.5`, `jamesgpt1/sf_model_e5`, `thenlper/gte-base`. We thus evaluated only the results retrieved by these models in the following. 

We initially observed a significant overlap in the documents retrieved by the three models. Since our aim was to identify the best model among the three, we were interested in their distinctive capabilities and therefore considered only results that differed between them. Each team member independently provided subjective evaluations of the retrieved results, ignoring their order and blind to the assessments of the others. We adopted a scoring system where 1 signified "not relevant", 2 indicated "neutral", and 3 denoted "relevant". These individual scores were then collated and averaged. For insights into the rationale behind our scoring, please see the annotated comments in [`correct_retrieval.jpg`](project/evaluation/retrieval_evaluation/qualitative_evaluation/images/correct_retrieval.jpg).

Interestingly, despite `thenlper/gte-base` dominating in the quantitative assessment, here, `BAAI/bge-base-en-v1.5` and `jamesgpt1/sf_model_e5` also demonstrated superior performance in some cases. RESULTS ...

In a next step, we investigated the order of the retrieved results and chose for each query the model that had the most effective ordering of the top three documents. Here, our attention was mainly on the firstly and secondly retrieved documents. We aggregate the scores for each model based on the frequency with which it was perceived as the best in terms of the order in which it presented its retrieval results. 

For our final decision of the embedding models, we carefully balanced the qualitative assessment with the precision of the quantitative analysis. We believe that our human perception is as important as the computational accuracy of the models we assess. This is reflected in the importance we place on our practical qualitative assessments. Such evaluations are especially important considering that our goal is to develop a chatbot that serves as a QA assistant and is ultimately judged on its ability to provide answers that are perceived effective and satisfactory by its human users.

On the quantitative side, we applied the inverse hyperbolic tangent function to all scores, ranging from 0 to 1, to highlight those approaching 1 and thus emphasize exemplary performance over the others. These recalibrated scores are then combined into a single metric and scaled by a factor of ten to fit the qualitative data.

🥇 After carefully looking at everything, we add up the best scores — the ones we highlighted in bold blue — to get our final score. By taking all these steps, we’ve decided that `thenlper/gte-base` is our top pick for the best model.

<img src="evaluation/retrieval_evaluation/qualitative_evaluation/images/evaluation_results.png" width="1000" />


### <a name="chatmodel-eval"></a>b) Evaluation of Chatmodel

We evaluated the various parameters and configurations of our model(s)...

To ensure consistency in our evaluation, we selected a set of 10 evaluation questions: 5 were randomly chosen from our QA evaluation dataset, covering a range of medical fields. The other 5 were generated by ChatGPT-4, prompted to create general questions on intelligence topics based on an initial list of 50 questions from the QA dataset. We lastly added 2 unrelated questions about movies and football. This mix aims to assess the model's ability to handle diverse medical topics and its tendency to hallucinate or acknowledge gaps in its knowledge. The questions can be found [here](evaluation/llm_evaluation/EVAL_QUESTIONS.md).

#### I. Prompt Engineering

**Setup of the Experiment:** An important step in the construction of a chatbot with RAG is to choose a proper prompt template. Sometimes even very similar templates can lead to different outputs in terms of quality, completeness and risk of hallucination. It is also crucial to make the tests on a specific model, since the best results on one model do not imply a good result on another one. We tested therefore the prompts on our final choice: `mistralai/Mistral-7B-Instruct-v0.1`.

We manually selected seven prompts, detailed in [`perform_prompt_tests.ipynb`](evaluation/llm_evaluation/prompt_engineering/perform_prompt_tests.ipynb) and designed to guide the generation of concise and relevant answers based on provided context for question-answering tasks while preventing hallucination.

Two examples are:

```py
text_prompt_2 = \
"""
You are an assistant for question-answering tasks.
Use the following pieces of retrieved context to answer the question.
If you don't know the answer, just say that you don't know.
Use three sentences maximum and keep the answer concise.
Question: {question}
Context: {context}
Answer:
"""
```

```py
text_prompt_3_short = \
"""
Context information is below.
{context}
Given the context information and not prior knowledge, answer the query.
Query: {question}
Use maximum three sentences.
Answer:
"""
```

In general, the "`_short`" version of a template is identical to the base version, which contains an additional request of concision.

In a thorough evaluation, we ran the 17 different queries selected for this purpose (as described above) on all of them. The answers were orderly formatted in markdown and then manually analyzed.

**Analysis of the Outcomes:** With so much data we needed a structured way to collect and categorize the outcomes. For this task we built a long table (see [`interpretation_results_promt_engineering.xlsx`](evaluation/llm_evaluation/prompt_engineering/results/interpretation_results_promt_engineering.xlsx)) where for every answer of every prompt template we annotated if they had any shortcomings. For the first 15 queries, the possible failures were:

1) Wrong or missing answer
2) Too short, too long or redundant
3) Presence of anomalies in the answer

For the last two queries we considered instead:
1) Hallucination
2) Redundancies (instead of a clear statement of not having the resources to answer the specific question)

Clearly the penalties were weighted differently depending on the severity. This will be clear in the presentation of the results.

<p align="left">
  <img src="./docs/images/prompt-engineering_table_analysis.png" width="700"/>
</p>

Moreover we noted and wrote down some trends in the quality and nature of the answers.
<p align="left">
  <img src="./docs/images/prompt-engineering_annotations.png" width="600"/>
</p>

**Interpretation of the Results:** Summing all the entries of the previous table, we obtained the following penalty scores:

<p align="left">
  <img src="./docs/images/prompt-engineering_penalty-scores.png" width="600"/>
</p>

As mentioned before, the severity of the shortcomings was manually assessed. The logic is that a prompt that gave seven long-winded answers is still less problematic than one that caused three wrong or missing answers. Combining this observations with our annotations we reduced our choice to three templates. The one which ended up being chosen (`tempalte_3-short`, which is among the two reported above) is a small bet on our side: it is by far the best at answering but it demonstrates some light tendencies to hallucinate. We always had a fallback in mind, with the second best and the third best templates, which are less punctual at responding, but are more solid and less prone to hallucinate, a safe bet.


#### II. Hybrid Search Weights

Through extensive testing with varying weights, we optimized the balance between term-specific accuracy and semantic understanding. We used the ten questions sampled from the QA dataset and evaluated the generated responses against the ground truth answers from the dataset. We computed BLEU, ROUGE and BERTScore to get a quantitative measure of similarity between generated and ground truth answer (see [`compute_bleu_rouge_bertscore.ipynb`](evaluation/llm_evaluation/hybrid_search/compute_bleu_rouge_bertscore.ipynb)). Our results indicate that the hybrid model, with equal weights of 0.5 for both keyword and vector search methods, showcases optimal effectiveness in addressing a broad spectrum of search needs

<p align="left">
  <img src="./docs/images/bleu_rouge_bert_hybrid_search.png" width="700"/>
</p>

#### III. Handling of Different Question Types

To assess how our chatbot manages various types of questions, we conducted a performance evaluation focusing on complex, causal, factoid, list, and hypothetical questions, as documented in [`question_types_evaluation.md`](evaluation/llm_evaluation/question_types/question_types_evaluation.md). We also compared the responses with those produced by ChatGPT-4 to discern differences in handling questions without specific context. The results are accessible [here](evaluation/llm_evaluation/question_types/testset_different_question_types.xlsx). 

ℹ️ This evaluation, tailored to the needs of our primary users — medical professionals — is inherently subjective. We examined the responses, highlighting our preferred ones in the results document with a dotted outline. 

💡 In 70% of instances, our chatbot was the source of the favored response. A notable distinction between our system and ChatGPT-4 lies in the response length. Unlike ChatGPT-4’s lengthy elaborations, our chatbot delivers concise answers, comprising essential information effectively. This approach prevents information overload, ensuring users receive precise and focused information. In contrast, ChatGPT-4 tends to offer more generalized and expansive answers, often providing more detail than necessary for the query at hand. One of the main advantages that becomes obvious here is of course the links to the specific documents used for the answer generation.

Below is an example of the differing answer types:

| Question | Answer of Our System | Answer of ChatGPT-4 |
|----------|----------------------|---------------------|
| *Is regular breakfast consumption associated with increased IQ in kindergarten children?* | ![](./evaluation/llm_evaluation/question_types/breakfast_answer_RAG.png) | ![](./evaluation/llm_evaluation/question_types/breakfast_answer_gpt.png) |


<!-- 
## 4.3 Experimental Details
- explain configurable parameters of our methods
- explain specific usage 
-->

<!-- 
## 4.4 Results
- compare own results to baseline
    - use basic chatmodel as baseline (maybe the one used in one of the assignments) and compare it with our choice
    - idea: 10 questions give to ChatGPT and our system: does RAG improve performance/prevent hallucinations
- present plots/tables of the before explained evaluation
-->

## <a name="analysis"></a>4.4 🧐 Analysis
<!-- 
- present qualitative analysis
- does our system work as expected?
- cases in which system consistently succeeds/fails? 
- does baseline succeeds/fails in same cases?
- use examples & metrics to underline our points instead of stating unproven points
-->

# <a name="limitations-results"></a>5. ⚡️ Limitations & Future Work 🔮

During the span of this project we had many itneresting ideas, but, unfrotunately, due to time and resources' constrains, we couldn't implement and try out all of them. Moreover, we are aware of some limitations in this release. Here we want to openly recognise them, discuss them and introduce the reader to some possible hypothetical solutions we devised. If we had a chance to work on thsis project again in the future, they would be surely topics we would address. At the same time, they can be inspiration for future work if someone else wanted to pick up our pjocect for further developement in the future.

#### I. Token Limits

There are several token limits to consider. We efficiently handled the input token limit via chunking our documents. The context length token limit, however, is a bit more challenging. For our LLM, the maximum `context_len` is 8192 tokens ([Jiang et al., 2023](#mistral)). We currently reach this limit in longer conversations after a few questions. In the future, we would consider different strategies to save tokens and surpass this limitation:

- Shortening the prompt message and retrieved documents by deleting unnecessary words such as function words. 

- Creating summaries of retrieved documents and/or of the whole chat history. This could be done after every query or once the limit is reached, a summary of the whole conversation could be given as input to a new conversation. This solution requires carful consideration of suitable summary models.

- Multithreading via [lightspeedGPT](https://github.com/andrewgcodes/lightspeedGPT) could be used to segment longer documents or chat history and process them in parallel. Answers would be collected and then assembled to a single answer. This would of course entail consideration about how to properly combine the multiple generated answers.

Since this limitation was not a priority for us, we therefore limited ourselves for the time being to reducing the context provided to two documents each, which are retrieved by the keyword search and the vector search. For future work, however, we find the aforementioned solutions interesting and can imagine that they will lead to a more competitive chatbot that can be used in real-world scenarios.

### II. Chain of Thought Prompting

Currently we are using a rather simple prompt, but in the future we would investigate the possibility of a Chain of Thought prompting. This instructs the LLM to solve a complex question step by step by making intermediate steps ("chains of thought") explicit. Instructions such as "Let's think step by step" are used to guide the model through the thought process. This can be particularly helpful for complex queries that require a step-by-step way of thinking. It should be stated that this kind of prompting can lead to longer answers, as the thought process is also output by the model, and could thus lead to reaching token limits. Moreover, the correctness of the thought process is not garantueed: while the model can generate a logically appearing thought process, it can still arrive at an incorrect conclusion. Lastly, this method is only useful for complex tasks and not required for simple yes/no questions. 

#### III. Inclusion of Metadata

Our chatbot does not currently make use of the rich ammount of metadata coming with the abstracts, most notably the authors' names and the publication dates. It was not our intention to inctroduce in the app interface hard coded boxes to filter on these kind of requirements, as we wanted our product to be a simple chatbot rather than a search engine. Therefore we could only rely on capturing the metadata from the user's questions. For future work however, we would consider the following solutions:

- We could modify the prompts so that the chatbot responded differently when the user asks a question related to metadata. For instance, _"What were the developments made in 2006 for the cure of cancer?"_ 

- Alternatively, we could implement a system where users could specify in the query to perform metadata filtering such as: _"[year=2006] What are the developments made in 2006 for the cure of cancer?"_ 

Unfortunately, the filtering criteria do not work well with BM25 and FAISS. In FAISS, the filtering occurs after retrieving the documents. For example, you can retrieve 1,000 documents and then apply the filtering criteria based on metadata. For BM25, there is no metadata-filtering support. In this context, Pinecone offers better support.

While we focused our attention on other issues, we recognise the great potential that this itegration could bring in the future.

#### IV. LLM Fine-Tuning
<span style="color:red"> **MISSING**</span>


#### V. Expansion to other Domains

Currently, our chatbot is limited to the biomedical field, more specifically to topics related to intelligence. However, a promising path lies ahead of us, as our RAG system can easily be expanded to include additional data from other fields. Adding new documents from the latest research is also theoretically straightforward. While we are currently limited to English documents, there may also be an extension to other languages in the future.

# <a name="conclusion"></a>6. 💡 Conclusion
<!-- 
- recap briefly main contributions
- highlight achievements
- reflect limitations
- outline possible extensions of our system or improvements for the future
- briefly give insights what was learned during this project
-->

***

# <a name="references"></a>7. References

- <a name="RRF"></a>Cormack, Gordon V., Clarke, Charles L. A., & Buettcher, Stefan. (2009). Reciprocal Rank Fusion Outperforms Condorcet and Individual Rank Learning Methods. *Proceedings of the 32nd International ACM SIGIR Conference on Research and Development in Information Retrieval (SIGIR '09)*, 758–759. Association for Computing Machinery, New York, NY, USA. [https://doi.org/10.1145/1571941.1572114](https://doi.org/10.1145/1571941.1572114)

- <a name="LDA"></a>Blei, David M., Ng, Andrew Y. & Jordan, Michael I. (2003). Latent Dirichlet Allocation. *Journal of Machine Learning Research*, 3, 993–1022. [https://www.jmlr.org/papers/volume3/blei03a/blei03a.pdf](https://www.jmlr.org/papers/volume3/blei03a/blei03a.pdf)

- <a name="mistral"></a>Jiang, Albert Q., Sablayrolles, Alexandre, Mensch, Arthur, Bamford, Chris, Chaplot, Devendra Singh, de las Casas, Diego, Bressand, Florian, Lengyel, Gianna, Lample, Guillaume, Saulnier, Lucile, Lavaud, Lélio Renard, Lachaux, Marie-Anne, Stock, Pierre, Le Scao, Teven, Lavril, Thibaut, Wang, Thomas, Lacroix, Timothée & El Sayed, William. (2023). Mistral 7B. [https://arxiv.org/pdf/2310.06825.pdf](https://arxiv.org/pdf/2310.06825.pdf)

- <a name="stopwords"></a>Miyajiwala, Aamir, Ladkat, Arnav, Jagadale, Samiksha & Joshi, Raviraj. (2022). On Sensitivity of Deep Learning Based Text Classification Algorithms to Practical Input Perturbations. *Intelligent Computing*, 613–626. Springer International Publishing. [https://doi.org/10.1007/978-3-031-10464-0_42](https://doi.org/10.1007/978-3-031-10464-0_42)

- <a name="TSDAE"></a>Wang, Kexin, Reimers, Nils & Gurevych, Iryna. (2021). TSDAE: Using Transformer-based Sequential Denoising Auto-Encoder for Unsupervised Sentence Embedding Learning. [https://arxiv.org/abs/2104.06979](https://arxiv.org/abs/2104.06979)



# <a name="appendix"></a>8. 💻 Appendix

## <a name="contributions"></a>8.1 🧑🏻‍🎓 Contributions

Please refer to [Asana](https://app.asana.com/0/1206188541316840/1206194377445034), the task manager we used over the course of all the project. All the tasks are unpacked and are labeled with whom was in charge to complete them. The access was granted to our supervisor during the project. 

We would like to specify that the group had a good chemistry and we all worked together to the final goal, helping each other out and coordinating efficiently when some tasks were dependent on others.

## <a name="anti-plagiarism"></a>8.2 📝 Anti-Plagiarism Confirmation

<p align="left">
  <img src="./docs/images/AntiPlagiat.jpg" width="700" />
</p>