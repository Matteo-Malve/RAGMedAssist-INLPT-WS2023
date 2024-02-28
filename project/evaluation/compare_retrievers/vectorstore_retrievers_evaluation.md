# Vectorstore Retrievers Evaluation

This document presents the evaluation results of two vectorstore retrievers: Faiss and Pinecone, over a range of top `k` values with the search type similarity.

## Evaluation Results

The evaluation was conducted over a set of 167 queries, comparing the performance based on the following metrics:
- Execution time per query
- Total execution time
- Match count
- Success percentage

###  Results for FAISS

| Metric | Execution Time per Query | Total Execution Time | Match Count | Success Percentage |
|--------|--------|-------|--------|-------|
| k=1 | 0.0211 seconds | 3.5279 | 152 | 91.02% |
| k=2 | 0.0209 seconds | 3.4897 | 158 | 94.61% |
| k=3 | 0.0207 seconds | 3.4646 | 163 | 97.60% |
| k=4 | 0.0207 seconds | 3.4566 | 163 | 97.60% |
| k=5 | 0.0208 seconds | 3.4670 | 164 | 98.20% |

###  Results for Pinecone

| Metric | Execution Time per Query | Total Execution Time | Match Count | Success Percentage |
|--------|--------|-------|--------|-------|
| k=1 | 0.2124 seconds | 35.4665 | 152 | 91.02% |
| k=2 | 0.2247 seconds | 37.5285 | 159 | 95.21% |
| k=3 | 0.2227 seconds | 37.1911 | 162 | 97.01% |
| k=4 | 0.2232 seconds | 37.2780 | 163 | 97.60% |
| k=5 | 0.2248 seconds | 37.5450 | 164 | 98.20% |

## Performance Comparison

The performance comparison between FAISS and Pinecone retrievers is visualized in the following sections.
### Execution Time per Query
![Execution Time per Query Plot](images/execution_time_per_query_plot.png)

### Total Execution Time
![Total Execution Time Plot](images/total_execution_time_plot.png)

### Success Percentage
![Success Percentage Plot](images/success_percentage_plot.png)

