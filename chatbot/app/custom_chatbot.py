"""
Custom chatbot model for RAG.
"""

from dotenv import load_dotenv, find_dotenv
import os
import pickle
import time
import warnings
import yaml

import pinecone
import torch
from transformers import (
    AutoModel,
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    GenerationConfig,
    pipeline,
)

from langchain import HuggingFacePipeline, hub
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.retrievers import BM25Retriever, EnsembleRetriever
from langchain.text_splitter import (
    CharacterTextSplitter,
    RecursiveCharacterTextSplitter,
    TokenTextSplitter,
)
from langchain_community.vectorstores import Pinecone
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain_community.llms import GPT4All
from langchain.chains import ConversationalRetrievalChain
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.chains import TransformChain
from langchain.chains import SequentialChain
import logging
from chatbot.app.custom_retriever import CustomEnsembleRetriever

# Suppress warnings
warnings.filterwarnings("ignore")

print(f"cuda is available: {torch.cuda.is_available()}")
print(f"mps is available: {torch.backends.mps.is_available()}")

# Automatically find and load the .env file
load_dotenv(find_dotenv())

# Please put this key in .env, this line will be deleted in the future
os.environ["PINECONE_API_KEY"]="1218c885-67e3-492f-b1ab-215405569e97"


class MedicalChatbot:
    def __init__(self, cfg):
        self.ensemble_retriever = None
        self.llm = None
        self.device = torch.device("cuda" if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_built() else "cpu"
        )
        self.chat_history = []
        self.cfg = cfg

        if[self.cfg["use_qa_chain"] and type(self.cfg["use_qa_chain"]) == bool]:
            self.qa_chain = self.init_qa_chain()

        if[self.cfg["use_conversational_qa_chain"] and type(self.cfg["use_conversational_qa_chain"]) == bool]:
            self.conversational_qa_chain = self.init_conversational_qa_chain()
            self.conversational_chat_history = []

        if[self.cfg["use_multi_query_qa_chain"] and type(self.cfg["use_multi_query_qa_chain"]) == bool]:
            self.multi_query_qa_chain = self.init_multi_query_qa_chain()

    # ------------------------------------------------------------------------------------------------------------------
    # Models' loading
    # ------------------------------------------------------------------------------------------------------------------

    def load_embedding_model(self):
        model_kwargs = {"device": self.device}
        encode_kwargs = {"normalize_embeddings": True}
        return HuggingFaceEmbeddings(
            model_name=self.cfg["embedding_model"],
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs,
        )

    def load_bm25_retriever(self):
        chunk_path = self.cfg["retrievers"]["bm25"]["path"]
        topk = self.cfg["retrievers"]["bm25"]["topk"]
        with open(chunk_path, "rb") as file:
            chunked_docs = pickle.load(file)
        return BM25Retriever.from_documents(chunked_docs, k=topk)

    def load_retrievers(self, name):
        if name in ["faiss", "pinecone"]:
            self.dense_retriever =  self.load_db_retriever(name)
            return self.dense_retriever
        elif name == "bm25":
            return self.load_bm25_retriever()

    def load_db_retriever(self, retriever_name):
        if retriever_name == "pinecone":
            return self.load_pinecone_db_retriever()
        else:
            return self.load_faiss_db_retriever()

    def load_faiss_db_retriever(self):
        faiss_index_path = f"{self.cfg['retrievers']['faiss']['faiss_index_path']}{self.cfg['embedding_model']}"
        db = FAISS.load_local(faiss_index_path, self.load_embedding_model())
        return db.as_retriever(
            search_type=self.cfg['retrievers']['faiss']['search_type'],
            search_kwargs={"k": self.cfg["retrievers"]["faiss"]["topk"],
                           "score_threshold": self.cfg["retrievers"]["faiss"]["score_threshold"]},
        )

    def load_pinecone_db_retriever(self):
        pinecone_api_key = os.getenv("PINECONE_API_KEY")
        if not pinecone_api_key:
            raise ValueError(
                "Pinecone API key not found. Please set it in the .env file."
            )

        pc = pinecone.Pinecone()
        index = pc.Index(self.cfg["retrievers"]["pinecone"]["index_name"])
        db = Pinecone(index, self.load_embedding_model(), "text")
        # save retriever because we can need it to initilize other chains (conversationa)
        return db.as_retriever(
            search_type=self.cfg["retrievers"]["pinecone"]["search_type"],
            search_kwargs={"k": self.cfg["retrievers"]["pinecone"]["topk"],
                           "score_threshold": self.cfg["retrievers"]["faiss"]["score_threshold"]})


    def load_custom_ensemble_retriever(self):
        if self.ensemble_retriever is None:
            ensemble_list = []
            weights = self.cfg["ensemble"]["weights"]
            topk_rrf= self.cfg["ensemble"]["topk_rrf"]
            for name, value in self.cfg["retrievers"].items():
                retriever = self.load_retrievers(name)
                ensemble_list.append(retriever)

            if not ensemble_list:
                raise ValueError("No valid retrievers were loaded.")
            self.ensemble_retriever = CustomEnsembleRetriever(topk_rrf=topk_rrf, retrievers=ensemble_list, weights=weights)

        return self.ensemble_retriever

    def get_llm(self):
        if self.llm is None:
            if "GPT4ALL" in self.cfg['llm_model']['local_path'].upper():
                self.llm = GPT4All(model=self.cfg['llm_model']['local_path'],
                              # max_tokens=2048,
                              )
            else:
                self.llm = self.init_llm_pipeline()
        return self.llm

    # ------------------------------------------------------------------------------------------------------------------
    # Initializations
    # ------------------------------------------------------------------------------------------------------------------

    def init_model(self):
        quantization_config = None
        if self.device.type == "cuda":
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                # load_in_8bit=True,
                # load_in_8bit_fp32_cpu_offload=True,  # Enable CPU offloading
                # bnb_8bit_compute_dtype=torch.float32,  # Use FP32 for computation when offloaded
            )
        return AutoModelForCausalLM.from_pretrained(
            self.cfg['llm_model']['name'],
            torch_dtype=torch.float16 if self.device.type == "cuda" else "auto",
            # torch_dtype=torch.float32 if self.device.type == "cuda" else "auto",
            device_map="auto",
            trust_remote_code=True,
            quantization_config=quantization_config if quantization_config else None,
        )

    def init_tokenizer(self):
        return AutoTokenizer.from_pretrained(
            self.cfg["llm_model"]["name"], use_fast=True
        )

    def init_llm_pipeline(self):
        generation_config = GenerationConfig.from_pretrained(self.cfg["llm_model"]["name"])
        generation_config.max_new_tokens = 512
        generation_config.temperature = self.cfg["llm_model"]["temperature"]
        generation_config.top_p = self.cfg["llm_model"]["top_p"]
        generation_config.do_sample = True
        generation_config.repetition_penalty = self.cfg["llm_model"]["repetition_penalty"]

        pipe = pipeline(
            "text-generation",
            model=self.init_model(),
            tokenizer=self.init_tokenizer(),
            # device=None if self.device=="cuda" else self.device,
            generation_config=generation_config,
        )
        return HuggingFacePipeline(
            pipeline=pipe
        )

    def init_qa_chain(self):

        llm = self.get_llm()
        prompt = self.get_prompt()

        return RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=self.load_custom_ensemble_retriever(),
            return_source_documents=True,
            chain_type_kwargs={"prompt": prompt},
        )

    def init_conversational_qa_chain(self):

        llm = self.get_llm()
        retriever = self.load_custom_ensemble_retriever()

        return ConversationalRetrievalChain.from_llm(llm=llm,
                                                     verbose=True,
                                                     retriever=retriever,
                                                     return_source_documents=True,
                                                     )
    # ------------------------------------------------------------------------------------------------------------------
    # Multi-query specific functions
    # ------------------------------------------------------------------------------------------------------------------

    def init_multi_query_qa_chain(self):

        logging.basicConfig()
        logging.getLogger("langchain.retrievers.multi_query").setLevel(logging.INFO)

        multi_query_retriever = MultiQueryRetriever.from_llm(
            retriever=self.load_custom_ensemble_retriever(), llm=self.get_llm())

        multi_query_retrieval_transform_chain = self.load_multi_query_retrieval_transform_chain(multi_query_retriever)

        multi_query_qa_chain = LLMChain(llm=self.get_llm(), prompt=self.get_prompt())

        return SequentialChain(
            chains=[multi_query_retrieval_transform_chain, multi_query_qa_chain],
            input_variables=["question"],  # we need to name differently to output "query"
            output_variables=["query", "context", "source_documents", "text"]
        )

    def load_multi_query_retrieval_transform_chain(self, multi_query_retriever):

        def multi_query_retrieval_transform(inputs: dict) -> dict:
            retrieved_docs = multi_query_retriever.get_relevant_documents(query=inputs["question"])
            docs = [d.page_content for d in retrieved_docs]
            docs_dict = {
                "query": inputs["question"],
                "context": "\n---\n".join(docs)[:2048],  # context window is 2048
                "source_documents": retrieved_docs
            }
            return docs_dict

        return TransformChain(
            input_variables=["question"],
            output_variables=["query", "context", "source_documents"],
            transform=multi_query_retrieval_transform
            )

    # ------------------------------------------------------------------------------------------------------------------
    # Prompt template setup
    # ------------------------------------------------------------------------------------------------------------------

    def get_prompt(self):
        #template = """<s> [INST] You are a helpful assistant for biomedical question-answering tasks.
        #            Use only the following retrieved context to answer the question. If the answer is not in the context,
        #            just say that you don't know.
        #            Provide a response strictly based on the information requested in the query.[/INST] </s>
        #            [INST] Question: {question}
        #            Context: {context}
        #            Answer: [/INST]"""
        template = \
        """
        Context information is below.
        {context}
        Given the context information and not prior knowledge, answer the query.
        Query: {question}
        Use maximum three sentences.
        Answer:
        """
        return PromptTemplate.from_template(template)
        # return hub.pull("rlm/rag-prompt", api_url="https://api.hub.langchain.com")

    # ------------------------------------------------------------------------------------------------------------------
    # Response generation
    # ------------------------------------------------------------------------------------------------------------------


    def generate_response_by_type(self, user_query, type='basic', raw_response=False):
        if type == 'basic':
            return self.generate_response(user_query, raw_response)
        elif type == 'multi_query':
            return self.generate_response_with_multi_query(user_query, raw_response)
        elif type == 'conversational':
            return self.generate_response_with_conversational(user_query, raw_response)
        else:
            raise ValueError(f"Unsupported response type: {type}")

    def no_docs_response(self, user_query, return_raw=False):
        response = {"query": user_query, "result": "Sorry, but I don't know as my capabilities are focused on medical assistance", "source_documents":[]}
        self.chat_history.append(response)
        return self._generate_response(response, return_raw=return_raw)

    def check_no_docs_retrieved(self, user_query):
        retrieved_docs_count = len(self.dense_retriever.get_relevant_documents(user_query))
        return retrieved_docs_count == 0

    def generate_response(self, user_query, return_raw=False):
        if self.check_no_docs_retrieved(user_query):
            return self.no_docs_response(user_query, return_raw)

        response = self.qa_chain({"query": user_query})
        self.chat_history.append(response)
        return self._generate_response(response, return_raw=return_raw)

    def generate_response_with_multi_query(self, user_query, return_raw=False):
        if self.check_no_docs_retrieved(user_query):
            return self.no_docs_response(user_query, return_raw)

        response = self.multi_query_qa_chain({"question": user_query})
        response["result"] = response["text"]
        del response["question"], response["text"]  # unnecessary key-value
        self.chat_history.append(response)
        return self._generate_response(response, return_raw=return_raw)

    def generate_response_with_conversational(self, user_query, return_raw=False):
        if self.check_no_docs_retrieved(user_query):
            return self.no_docs_response(user_query, return_raw)

        max_history_length = self.cfg["conversational_chain"]["conversation_depth"]
        conversation_history = self.conversational_chat_history[-max_history_length:]
        response = self.conversational_qa_chain({"question": user_query, "chat_history": conversation_history})
        response["result"] = response["answer"]
        response["query"] = response["question"]
        del response["answer"], response["question"]  # use keys result, query instead answer, question
        self.chat_history.append(response)
        self.conversational_chat_history.append((user_query, response["result"]))
        return self._generate_response(response, return_raw=return_raw)


    # ------------------------------------------------------------------------------------------------------------------
    # Miscellaneous
    # ------------------------------------------------------------------------------------------------------------------

    def clean_chat_history(self):
        self.chat_history = []

    def set_similarity_score_threshold(self, score):
        self.dense_retriever.search_kwargs["score_threshold"] = score

    def retrieve_pmid_urls(self, response):
        response_documents = response.get('source_documents')
        if response_documents:
            base_url = "https://pubmed.ncbi.nlm.nih.gov/"
            pmid_set = set() # to avoid duplicates
            pmid_urls = []
            for document in response_documents:
                pmid = document.metadata.get('PMID')
                if pmid and pmid not in pmid_set:
                    pmid_set.add(pmid) # Add PMID to set to track uniqueness
                    pmid_urls.append(base_url + pmid)
            return pmid_urls

    def _generate_response(self, response, return_raw=False):
        """
        s.

        If `return_raw` is set to True, this function returns the raw response object directly.
        Otherwise, it formats the response as Markdown, enriching it with DOI links if available, to enhance the presentation.

        Parameters:
        - response (dict): The response object obtained from a query and contains 'query', 'result', 'source_documents'.
        - response_key (str): The key in the `response` dict that contains the actual response to be formatted generated by chain.
        - return_raw (bool): A flag to determine the format of the response. If True, the raw response object is returned;
          if False, a Markdown-formatted string including additional related links is returned.

        Returns:
        - If `return_raw` is True, the raw `response` object is returned.
        - Otherwise, a string formatted in Markdown that includes the response and any related DOI URLs is returned.
        """
        if return_raw:
            return response

        else:
            markdown_response = ""
            markdown_response += f"<p>{response['result']}</p>\n"
            pmid_urls = self.retrieve_pmid_urls(response)
            if pmid_urls:
                markdown_response += "<p>For more information, please refer to the following links:</p>\n"
                for url in pmid_urls:
                    markdown_response += f"<p><a href='{url}' target='_blank'>{url}</a></p>\n"
            return markdown_response
