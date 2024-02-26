from typing import List, Optional, cast
from langchain_core.runnables import RunnableConfig
from langchain_core.runnables.config import patch_config
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain.retrievers import EnsembleRetriever
from langchain_core.documents.base import Document


class CustomEnsembleRetriever(EnsembleRetriever):

    def rank_fusion(
        self,
        query: str,
        run_manager: CallbackManagerForRetrieverRun,
        *,
        config: Optional[RunnableConfig] = None,
    ) -> List[Document]:
        """
        Overridden method to perform rank fusion of documents retrieved by multiple retrievers.
        This implementation introduces special handling for scenarios involving a mix of retrievers,
        where exactly one retriever utilizes a similarity score threshold for filtering documents and
        other retrievers do not apply such thresholds. If the retriever with the similarity score threshold
        returns fewer documents than specified by its `search_kwargs["k"]` parameter (indicating that some top-k
        documents fall below the threshold), this method adjusts the document count from other retrievers to match,
        effectively aligning the result sets across different retrieval strategies.
        In all other cases, this method works the same as the official implementation for ensemble_retriever

        This adjustment addresses the limitation with sparse retrievers (e.g., BM25) on the langchain platform,
        which do not natively support filtering documents below a similarity threshold. By doing so, it ensures
        that the combined results from multiple retrievers are consistent in size and relevance, based on the
        specified threshold criteria.


        Retrieve the results of the retrievers and use rank_fusion_func to get
        the final result.

        Args:
            query: The query to search for.

        Returns:
            A list of reranked documents.
        """

        # Get the results of all retrievers.
        retriever_docs = [
            retriever.invoke(
                query,
                patch_config(
                    config, callbacks=run_manager.get_child(tag=f"retriever_{i+1}")
                ),
            )
            for i, retriever in enumerate(self.retrievers)
        ]

        # added part - START
        retriever_id_with_threshold = None
        for i, retriever in enumerate(self.retrievers):
            if hasattr(retriever, 'search_type'):
                if retriever.search_type == "similarity_score_threshold":
                    if retriever_id_with_threshold is None:
                        retriever_id_with_threshold = i
                    else:
                        retriever_id_with_threshold = None
                        break


        if retriever_id_with_threshold is not None:
            doc_count_above_threshold =  len(retriever_docs[retriever_id_with_threshold])
            if doc_count_above_threshold < self.retrievers[retriever_id_with_threshold].search_kwargs.get('k', 999):
                retriever_docs = [docs[:doc_count_above_threshold]for docs in retriever_docs]

        # added part - END

        # Enforce that retrieved docs are Documents for each list in retriever_docs
        for i in range(len(retriever_docs)):
            retriever_docs[i] = [
                Document(page_content=cast(str, doc)) if isinstance(doc, str) else doc
                for doc in retriever_docs[i]
            ]

        # apply rank fusion
        fused_documents = self.weighted_reciprocal_rank(retriever_docs)

        return fused_documents
