# [TITLE]
### Team members:
| Name and surname    |  Matric. Nr. | Course of study                            |   e-mail address   |
|:--------------------|:-------------|:-------------------------------------------|:-------------------|
| Matteo Malvestiti | 4731243| M.Sc. Data and Computer science (Erasmus) | matteo.malvestiti@stud.uni-heidelberg.de|
| Sandra Friebolin | 3175035 | M.Sc. Computational linguistics (?) | sandra_friebolin@proton.me |
| Yusuf Berber | ...... | M.Sc. Data and Computer Science (?) | yusuf.berber@stud.uni-heidelberg.de |

### Member contributions
Please refer to Asana, the task manager we used over the course of all the project. All the tasks are unpacked and are labeled with whom was in charge to complete them. The access was granted to our supervisor during the project. \
We will like to specify that the group had a good chemistry and we all worked together to the final goal, helping each other out and coordinating efficiently when some tasks were dependent on others.

### Advisor
Robin Khanna (R.Khanna@stud.uni-heidelberg.de)

### Anti-plagiarism Confirmation
.....


***
# 1. Introduction
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


# 2. Related Work
- put our work into context of current research
- including papers read for research/that used same techniques but applied to different problems
- emphasize how our work differs from previous work, outlining their limitations/why our application domain is different
- ⚠️ only major points, not too much detail


# 3. Approach
- conceptual details of our system (about its functionality, its components, data processing pipelines, algorithms, key methods)
- 💡 be specific about methods (include equations, show figures...)
- 💡 emphasize creative/novel parts, but also properly cite existing methods
- 💡 stick to fixed vocabulary (mathematical notations, method & dataset names) and writing style!
- describe baseline approaches (briefly if from external source)

## 3.1 Data Processing Pipelines

### 3.1.1 Data Pre-Processing
Several data cleaning and pre-processing strategies were considered and applied according to their usefulness to our specific application:

❌ **Lowercasing:** Converting all text to lowercase can be beneficial for consistency and improve performance for many embedding models. However, in certain scientific contexts, such as our biomedical texts, case sensitivity is important, for instance when distinguishing between gene names and common words. We thus decided against lowercasing, also given that we used very advanced embedding models able to handle case sensitivity.

❌ **Handling Stop Words:** Removing common words that do not contribute much meaning to the sentences was a common practice for traditional approaches as it gives more focus on important content words. However, advanced embedding models, especially those based on Transformer architecture, are designed to understand and generate natural language capturing the context of the entire sentence or document, including the function of stop words [add citation]. We thus decided against removing them.

❌ **Handling Bigrams or N-grams:** Advanced Transformer based models do not require this step since they are designed to capture word context using their attention mechanisms and positional embeddings, making explicit n-gram creation less necessary.

✅ **Removing Special Characters:** This includes stripping out unnecessary punctuation, symbols, or special characters that are not relevant to the analysis or could interfere with the model's understanding of the text. We apply this step to to enhance data consistency and reduce noise, thereby improving model focus and efficiency.

✅ **Normalization:** This process standardizes text to a consistent format, enhancing data uniformity and simplifying processing. Specifically, we implemented Unicode normalization to transform author names with special characters — common in languages such as Swedish and French — into a uniform representation. 

✅ **Tokenization:** Essential to the embedding process, tokenization divides text into manageable pieces, or tokens. Our chosen embedding models employ unique tokenizers that segment text into words and subwords. This granularity enables precise interpretation of biomedical terminology, accommodating the field's extensive vocabulary and specialized jargon.

✅ **Removing Short Abstracts:** Abstracts with fewer than 100 characters often lack sufficient detail, offering minimal insight. To enhance our dataset's quality, we excluded such brief abstracts and those lacking any abstract text. This refinement process resulted in the retention of 58,535 abstracts, effectively removing 319 from our initial collection.


## 3.2 Algorithms/Methods

We integrated Langchain's `EnsembleRetriever` into our search framework to make use of a hybrid model that combines BM25-based keyword search with vector search to provide precise and contextually relevant results. This approach is particularly beneficial for datasets dealing with highly specific terms, such as our biomedical abstracts, where keyword search excels in precision. By leveraging the strengths of both methodologies, we ensure users receive accurate information that not only aligns with their query's intent but also navigates the complexities of specialized terminology. 

## 3.3 Baselines



# 4. Experimental Setup and Results

## 4.1 Data
- describe the data ✅
- outline where, when, and how it was gathered 
- show insightful metrics
    - length (max, min, average) ✅
    - distribution of publications (frequency per year to understand how interest in the topic "intelligence" has grown over time) ✅
    - most common authors ✅
    - make a topic analysis to see the most common topics (from titles? and abstracts?)
    - readability/accessibility scores (e.g., Flesch-Kincaid) on abstracts to assess how accessible the information is to general audiences, crucial for RAG

The dataset comprises abstracts and associated metadata from medical articles sourced from [PubMed](https://pubmed.ncbi.nlm.nih.gov/?term=intelligence+%5BTitle%2Fabstract%5D&filter=simsearch1.fha&filter=years.2013-2023&sort=date), a free search engine for life sciences and biomedical literature, managed by the U.S. National Library of Medicine at the National Institutes of Health. To manage time and computational constraints, our focus is limited to abstracts published between 2013 and 2023 featuring the keyword "intelligence", totaling 58,854 documents.

The documents in the dataset follow a structured format typical of biomedical literature. Each document contains several key elements that have designated abbreviations:

- **PMID (PubMed Identifier):** A unique number assigned to each PubMed record, used for easy reference and retrieval.
- **Title (TI):** The title of the article.
- **Abstract (AB):** A brief summary of the research, methods, results, and conclusions. It's a crucial part of the document as it provides the essence of the research without the need to read the full article.
- **Authors (FAU and AU):** Lists the full names (FAU) and initials (AU) of the authors, along with their affiliations (AD), providing information about who conducted the research and their institutional backgrounds.
- **Date of Publication (DP):** Indicates when the article was published, which is important for understanding the timeliness and relevance of the research.
- **DOI (Digital Object Identifier - LID):** A unique alphanumeric string assigned to the document, providing a permanent link to its location on the internet.
- **Additional Information:** Includes various bibliographic details like journal name (JT), issue (IP), volume (VI), language of the article (LA), grant and funding information (GR), and publication type (PT). However, we do not plan to use these as metadata for our application.

The metadata selected for our project encompasses the authors, title, date, and DOI of each document, as illustrated in this data point example:

<p align="center">
  <img src="./docs/images/datapoint_example.png" width="700" />
</p>

This information not only aids in establishing the credibility and context of the research but also enables our system to link directly to the source in the answers it generates - an additional functionality of our system. The abstracts, serving as the core of our dataset, will be utilized by our retrieval system to identify and present the most pertinent document in response to user queries, thereby forming the basis for generating informed and accurate answers.

We extracted the data [explain/how/when where it was gathered ...]

Following the data preprocessing steps, we conducted an in-depth analysis to extract meaningful insights about our dataset. We discovered that abstract lengths show a wide range, with the shortest being 93 characters and the longest reaching 60,664 characters. The average abstract length stands at 1,504.78 characters. The histogram presented below illustrates the distribution of abstract lengths, highlighting how frequently each length occurs.

<p align="center">
  <img src="./docs/images/distribution_abstract_length_log_scale.png" width="700" />
</p>

Turning our attention to the publication frequency on the topic of "intelligence," we noted a growing interest over time. The barplot below visualizes this upward trend, clearly showing a year-over-year increase in the number of publications, which signals a growing engagement with the topic.

<p align="center">
  <img src="./docs/images/distribution_publications_over_time.png" width="700" />
</p>

The visual analysis of publications per author reveals a common trend: the majority of authors contribute fewer than 2 publications on average, highlighting a broad base of singular contributions within the field. Notably, the most prolific author has made an impressive 94 contributions.

<p align="center">
  <img src="./docs/images/distribution_authors_frequency_contribution.png" width="700" />
</p>

[Ian J Deary](https://www.research.ed.ac.uk/en/persons/ian-deary-2) stands out as the leading author, reflecting his extensive involvement in intelligence and cognitive aging research. The following visualization ranks the top 10 authors by their number of publications. Notably, 'Unknown' appears as the fourth-highest entry, signaling some unidentified authors within the dataset.


<p align="center">
  <img src="./docs/images/distribution_authors_most_frequent.png" width="700" />
</p>


For analyzing common topics appearing in our dataset based on titles of publications, Latent Dirichlet Allocation (LDA) is a commonly used and simple method. It is a generative statistical model that allows sets of observations to be explained by unobserved groups that explain why some parts of the data are similar [add citation]. [... interpretation?]

<p align="center">
  <img src="./docs/images/topics_LDA.png" width="700"/>
</p>



## 4.2 Evaluation Method
- explain & define used/own metrics 
- motivate expected achievements

### 4.2.1 Evaluation of Information Retrieval 

We made use of the [PubMedQA](https://pubmedqa.github.io) for evaluating our retrieval method. This dataset contains [1,000 expert-labeled questions](https://github.com/pubmedqa/pubmedqa/blob/master/data/ori_pqal.json) together with both long and short answers, such as "yes/no", as well as the context and PubMed ID. Unfortunately, only 176 instances from our "Intelligence 2013-2023" dataset we use for context retrieval are contained in this evaluation dataset. We use these instances to compute accuracy, F1 score, mean reciprocal rank (MRR), and normalized discounted cumulative gain (nDCG) for our quantitative experiments, comparing the PMID of our retrieved documents with the ground truth PMID. 

**Accuracy:** We considered different values of top k results. The best three performing models for k=3 (we retrieve 3 documents as context for our chat model) are `thenlper_gte-base`, `BAAI_bge-base-en-v1.5` and `jamesgpt1_sf_model_e5`. Keyword search using BM25 was used as a baseline for comparison against our semantic search methods.

|                                  |   k=1 |   k=2 |   k=3 |   k=4 |   k=5 |   k=6 |   k=7 |   k=8 |   k=9 |   k=10 |   k=11 |   k=12 |   k=13 |   k=14 |   k=15 |   k=16 |   k=17 |   k=18 |   k=19 |   k=20 |
|:---------------------------------|------:|------:|------:|------:|------:|------:|------:|------:|------:|-------:|-------:|-------:|-------:|-------:|-------:|-------:|-------:|-------:|-------:|-------:|
| BM25              | 0.635  | 0.707 | 0.731  | 0.754 | 0.772 | 0.802 | 0.808 | 0.826  | 0.832  |  0.844  |  0.85  |  0.856 |  0.862 |  0.862 |  0.862 |  0.862 |  0.862 |  0.862 |  0.868 |  0.868 |
| dmis-lab_biobert-base-cased-v1.1 | 0.084 | 0.114 | 0.168 | 0.192 | 0.198 | 0.204 | 0.216 | 0.24  | 0.251 |  0.257 |  0.275 |  0.287 |  0.287 |  0.287 |  0.293 |  0.299 |  0.299 |  0.299 |  0.317 |  0.323 |
| all-MiniLM-L6-v2                 | 0.683 | 0.838 | 0.856 | 0.88  | 0.898 | 0.928 | 0.934 | 0.94  | 0.94  |  0.946 |  0.952 |  0.958 |  0.964 |  0.964 |  0.97  |  0.976 |  0.976 |  0.976 |  0.976 |  0.976 |
| **BAAI_bge-base-en-v1.5**            | 0.85  | 0.94  | **0.964** | 0.976 | 0.982 | 0.982 | 0.982 | 0.982 | 0.982 |  0.982 |  0.982 |  0.982 |  0.982 |  0.988 |  0.988 |  0.988 |  0.988 |  0.988 |  0.988 |  0.988 |
| llmrails_ember-v1                | 0.85  | 0.934 | 0.958 | 0.964 | 0.97  | 0.976 | 0.982 | 0.982 | 0.982 |  0.982 |  0.982 |  0.982 |  0.982 |  0.982 |  0.982 |  0.982 |  0.982 |  0.982 |  0.982 |  0.982 |
| **jamesgpt1_sf_model_e5**            | 0.856 | 0.922 | **0.964** | 0.97  | 0.97  | 0.982 | 0.982 | 0.982 | 0.982 |  0.982 |  0.982 |  0.982 |  0.982 |  0.982 |  0.982 |  0.982 |  0.982 |  0.982 |  0.982 |  0.982 |
| **thenlper_gte-base**                | 0.91  | 0.946 | **0.976** | 0.976 | 0.982 | 0.994 | 0.994 | 0.994 | 0.994 |  0.994 |  0.994 |  0.994 |  0.994 |  0.994 |  0.994 |  0.994 |  0.994 |  0.994 |  0.994 |  0.994 |
| intfloat_e5-base-v2              | 0.79  | 0.904 | 0.94  | 0.958 | 0.958 | 0.964 | 0.964 | 0.97  | 0.97  |  0.97  |  0.97  |  0.976 |  0.982 |  0.982 |  0.982 |  0.982 |  0.982 |  0.982 |  0.982 |  0.988 |

**F1 Score:**

**MRR:**

**nDCG:**

#### I. Quantitative Evaluation 



#### II. Qualitative Evaluation

### 4.2.2 Evaluation of Chatmodel

We evaluated the various parameters and configurations of our model(s)...

To ensure consistency in our evaluation, we selected a set of 10 evaluation questions: 5 were randomly chosen from our QA evaluation dataset, covering a range of medical fields. The other 5 were generated by ChatGPT-4, prompted to create general questions on intelligence topics based on an initial list of 50 questions from the QA dataset. We lastly added 2 unrelated questions about movies and football. This mix aims to assess the model's ability to handle diverse medical topics and its tendency to hallucinate or acknowledge gaps in its knowledge. The questions can be found [here](evaluation/llm_evaluation/EVAL_QUESTIONS.md).

### 4.2.3 Prompt engineering
#### Setup of the experiment
An important step in the construction of a chatbot with RAG is to choose a proper prompt template. Sometimes even very similar templates can lead to different outputs in terms of quality, completeness and risk of hallucination.
It's also crucial to make the tests on a specific model, since the best results on one model doesn't imply a good result on another one.
We tested therfore the prompts on our final choice: mistral.

First of all we manually selected seven prompts, which you can read in the notebook dedicated to this evalaution process, namely:

    > project/evaluation/llm_evaluation/prompt_engineering/perform_prompt_tests.ipynb

An example is:

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
or

    text_prompt_3_short = \
    """
    Context information is below.
    {context}
    Given the context information and not prior knowledge, answer the query.
    Query: {question}
    Use maximum three sentences.
    Answer:
    """
where, in general, the "_short" version of a template is the identical version of the base one with an additional request of concision. 
We evaluated all these template as thoroughly as possible.
Specifically, we ran 17 different queries on all of them:
- 10 queries from the Q&A dataset [previously mentioned in the IR qualitative evaluation]
- 5 questions generated by GPT-4
- 2 handcrafted queries to asses hallucination:

      "What are your favorite comedies?"
      "Which football team do you think will win the world cup this year?"

The answers were orderly formatted in markdown and then manually analyzed.

#### Analysis of the outcomes
With so much data we needed a structured way to collect and cathegorize the outcomes. For this task we built a long table where for every answer of every prompt template we annotated if they had any shortcomings. For the first 15 queries the possible failures were: 1) Wrong or missing answer; 2) Too short, too long or redundant; 3) Presence of anomalies in the answer.
For the last two queries instead: 1) Hallucination, 2) Redundancies, instead of a clear statement of not having the resources to answer the specific question.
Clearly the penalties were weighted differently depending on the severity. This will be clear in the presentation of the results.

<p align="center">
  <img src="./docs/images/prompt-engineering_table_analysis.png" width="700"/>
</p>

Moreover we noted and wrote down some trends in the quality and nature of the answers.
<p align="center">
  <img src="./docs/images/prompt-engineering_annotations.png" width="600"/>
</p>

#### Interpretation of the results

Summing all the entries of the previous table we obtained the following penalty scores:
<p align="center">
  <img src="./docs/images/prompt-engineering_penalty-scores.png" width="600"/>
</p>
As forementioned, the severity of the shortcomings was manually assessed. The logic is that a prompt that gave 7 long-winded answers is still less problematic than one that caused 3 wrong or missing answers.
Combining this observations with our annotations we reduced our choice to three templates. The one who ended up being chosen (tempalte_3-short, which is among the two reported above) is a small bet on our side: it's by far the best at answering but it demonstrates some light tendencies to hallucinate.
We always had a fallback in mind, with the second best and the third best templates, which are less punctual at responding, but are more solid and less prone to hallucinate, a safe bet.


#### Which LLM?

#### Which Prompt Works Best?

#### Hybrid Search Weights

Through extensive testing with varying weights, we optimized the balance between term-specific accuracy and semantic understanding. We used the ten questions sampled from the QA dataset and evaluated the generated responses against the ground truth answers from the dataset. We computed BLEU, ROUGE and BERTScore to get a quantitative measure of similarity between generated and ground truth answer. Our results indicate that the hybrid model, with equal weights of 0.5 for both keyword and vector search methods, showcases optimal effectiveness in addressing a broad spectrum of search needs

<p align="center">
  <img src="./docs/images/bleu_rouge_bert_hybrid_search.png" width="400"/>
</p>


<!-- TO DO: qualitative evaluation
-->



## 4.3 Experimental Details
- explain configurable parameters of our methods
- explain specific usage 

## 4.4 Results
- compare own results to baseline
    - use basic chatmodel as baseline (maybe the one used in one of the assignments) and compare it with our choice
    - idea: 10 questions give to ChatGPT and our system: does RAG improve performance/prevent hallucinations
- present plots/tables of the before explained evaluation

## 4.5 Analysis
- present qualitative analysis
- does our system work as expected?
- cases in which system consistently succeeds/fails? 
- does baseline succeeds/fails in same cases?
- use examples & metrics to underline our points instead of stating unproven points

# 5. Conclusion and Future Work
- recap briefly main contributions
- highlight achievements
- reflect limitations
- outline possible extensions of our system or improvements for the future
- briefly give insights what was learned during this project

***
# References






