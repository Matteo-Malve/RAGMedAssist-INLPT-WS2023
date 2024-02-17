"""
Custom chatbot model for RAG.
"""

from dotenv import load_dotenv, find_dotenv
import os
import pickle
import random
import time
import warnings
import yaml


import pandas as pd
import pinecone
import torch
from IPython.display import Markdown, display
from sentence_transformers import SentenceTransformer
from transformers import (
    AutoModel,
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    GenerationConfig,
    pipeline,
)


from langchain import HuggingFacePipeline, hub
from langchain.chains import RetrievalQA
from langchain_community.retrievers import BM25Retriever, EnsembleRetriever
from langchain.text_splitter import (
    CharacterTextSplitter,
    RecursiveCharacterTextSplitter,
    TokenTextSplitter,
)
from langchain.vectorstores import Pinecone
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate

# Suppress warnings
warnings.filterwarnings("ignore")


# Automatically find and load the .env file
load_dotenv(find_dotenv())



class MedicalChatbot:
    def __init__(self,cfg):
        self.device = torch.device(
            "cuda" if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_built() 
            else "cpu"
        )
        self.chat_history = []
        self.cfg = cfg
        self.qa_chain = self.init_qa_chain()

    def generate_response(self, user_query):
        response = self.qa_chain({"query": user_query})
        # self.chat_history.append({"query": user_query, "response": response})
        self.chat_history.append(response)
        return response

    def init_embedding_model(self):
        model_kwargs = {"device": self.device}
        encode_kwargs = {"normalize_embeddings": True}
        self.embed_model = HuggingFaceEmbeddings(
            self.cfg['embedding_model'],
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs,
        )

    def load_bm25_retriever(self):
        chunk_path = self.cfg['retriever']['bm25']['path']
        topk = self.cfg['retriever']['bm25']['topk']
        with open(chunk_path, "rb") as file:
            chunked_docs = pickle.load(file)
        return BM25Retriever.from_documents(chunked_docs, k=topk)

    def load_retrievers(self, name):
        if name in ['faiss', 'pinecone']:
            return self.load_db_retriever()
        elif name == "bm25":
            return self.load_bm25_retriever()

    def load_db_retriever(self, retriever_name, faiss_indices_path):
        if retriever_name == "pinecone":
            return self.load_pinecone_db_retriever()
        else:
            return self.load_faiss_db_retriever(faiss_indices_path)

    def load_faiss_db_retriever(self):
        faiss_index_path = f"{self.cfg['retriever']['faiss']['faiss_indices_path']}{self.cfg['embedding_model']}"
        db = FAISS.load_local(faiss_index_path)
        return db.as_retriever(
            search_type="similarity",
            search_kwargs={"k": self.cfg['retriever']['faiss']['topk']},
        )

    def load_pinecone_db_retriever(self):
        pinecone_api_key = os.getenv("PINECONE_API_KEY")
        if not pinecone_api_key:
            raise ValueError(
                "Pinecone API key not found. Please set it in the .env file."
            )
        pc = pinecone.init(api_key=pinecone_api_key)
        index = pc.Index(self.cfg["retriever"]["pinecone"]["index_name"])
        db = Pinecone(index, self.embed_model, "text")
        return db.as_retriever(search_type=self.cfg["retriever"]["pinecone"]["search_type"], 
                               search_kwargs={"k": self.cfg["retriever"]["pinecone"]["topk"]})

    def load_ensemble_retriever(self):
        ensemble_list = []
        for retriever in self.cfg["retrievers"]:
            ensemble_list.append(self.load_retrievers(retriever))

        return EnsembleRetriever(
            retrievers=ensemble_list, weights=self.cfg["ensemble"]["weights"]
        )

    def init_model(self):
        quantization_config = None
        if self.device.type == "cuda":
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
        return AutoModelForCausalLM.from_pretrained(
            self.cfg['llm_model']['name'],
            torch_dtype=torch.float16 if self.device.type == "cuda" else "auto",
            device_map="auto",
            trust_remote_code=True,
            quantization_config=quantization_config if quantization_config else None,
        )

    def init_tokenizer(self):
        return AutoTokenizer.from_pretrained(self.cfg['llm_model']['name'], use_fast=True)

    def init_llm_pipeline(self):
        generation_config = GenerationConfig.from_pretrained(
            self.cfg['llm_model']['name'],
            max_new_tokens=512,
            temperature=self.cfg['llm_model']['temperature'],
            top_p=self.cfg['llm_model']['top_p'],
            do_sample=True,
            repetition_penalty=self.cfg['llm_model']['repetition_penalty'],
        )
        pipe = pipeline(
            "text-generation",
            model=self.init_model(),
            tokenizer=self.init_tokenizer(),
            device=self.device,
            generation_config=generation_config,
        )
        return HuggingFacePipeline(pipe)

    def init_qa_chain(self):
        llm = self.init_llm_pipeline()
        prompt = self.get_prompt()
        return RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=self.load_ensemble_retriever(),
            return_source_documents=True,
            chain_type_kwargs={"prompt": prompt},
        )
    
    def get_prompt(self):
        #return hub.pull("rlm/rag-prompt", api_url="https://api.hub.langchain.com")
        return {"prompt":PromptTemplate.from_template(self.cfg['prompt_template'])}


    def clean_chat_history(self):
        self.chat_history = []



if __name__ == "__main__":

    with open("cfg.yaml", "r") as file:
        cfg = yaml.safe_load(file)

    chatbot = MedicalChatbot(cfg)
    response = chatbot.generate_response(str(input()))
    print(response)