import streamlit as st

# Streamlit app
st.title("Pubmed chat")

# Loading bar
st.write("Loading ...")
bar = st.progress(0)
progress_status=st.empty()

from streamlit_chat import message
import tempfile
import random
import torch
import os
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter, TokenTextSplitter, CharacterTextSplitter
bar.progress(10)
progress_status.write(str(10)+"%")
from langchain_community.vectorstores import FAISS
import pandas as pd
from transformers import BitsAndBytesConfig, AutoModelForCausalLM, AutoTokenizer, GenerationConfig, pipeline
import torch
from langchain import HuggingFacePipeline
from langchain.chains import RetrievalQA
from langchain import hub
from IPython.display import Markdown, display
import warnings
warnings.filterwarnings('ignore')
bar.progress(20)
progress_status.write(str(20)+"%")

class MedicalChatbot:
    def __init__(self, embedding_model_name='thenlper/gte-base', llm_model_name="mistralai/Mistral-7B-Instruct-v0.1", faiss_indices_path="../data/faiss_indices"):
        self.embedding_model_name = embedding_model_name
        self.llm_model_name = llm_model_name
        self.faiss_indices_path = faiss_indices_path
        self.device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_built() else "cpu"
        # self.init_embedding_model()
        # self.load_FAISS_DB_retriever()
        # self.init_tokenizer()
        # self.init_model()
        # self.init_pipeline()
        # self.init_llm()
        # self.init_prompt_template()
        # self.init_qa_chain()


    def generate_response(self, user_query):
        result = self.qa_chain({"query": user_query})
        return result

    def init_embedding_model(self):
        model_kwargs = {'device': self.device}
        encode_kwargs = {'normalize_embeddings': True}
        self.embed_model = HuggingFaceEmbeddings(model_name=self.embedding_model_name, model_kwargs=model_kwargs,
                                            encode_kwargs=encode_kwargs)

    def load_FAISS_DB_retriever(self):
        self.faiss_db = FAISS.load_local("../data/faiss_indices/thenlper/gte-base", self.embed_model)
        self.faiss_retriever = self.faiss_db.as_retriever(search_type="similarity", search_kwargs={"k": 3})


    def get_relevant_documents_for_query(self, query):
        return self.faiss_retriever.invoke(query)

    def init_model(self):
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True
        )

        self.model=AutoModelForCausalLM.from_pretrained(
            self.llm_model_name, torch_dtype=torch.float16,
            trust_remote_code=True,
            device_map="auto",
            quantization_config=quantization_config
        )

    def init_tokenizer(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.llm_model_name, use_fast=True)
        # self.tokenizer.pad_token = self.tokenizer.eos_token

    def init_pipeline(self):
        generation_config = GenerationConfig.from_pretrained(self.llm_model_name)
        generation_config.max_new_tokens = 512
        generation_config.temperature = 0.001
        generation_config.top_p = 0.95
        generation_config.do_sample = True
        generation_config.repetition_penalty = 1.15

        self.pipeline = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            return_full_text=True,
            generation_config=generation_config,
    )

    def init_llm(self):
        self.llm = HuggingFacePipeline(
            pipeline=self.pipeline,
    )


    def init_prompt_template(self):

        # template = """Use the following pieces of context to answer the question at the end.
        # If you don't know the answer, just say that you don't know, don't try to make up an answer.
        # Use three sentences maximum and keep the answer as concise as possible.
        # Always say "thanks for asking!" at the end of the answer.
        #
        # {context}
        #
        # Question: {question}
        #
        # Helpful Answer:"""
        # self.custom_rag_prompt = PromptTemplate.from_template(template)

        self.prompt  = hub.pull("rlm/rag-prompt", api_url="https://api.hub.langchain.com")

    def init_qa_chain(self):
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.faiss_retriever,
            return_source_documents=True,
            chain_type_kwargs={"prompt": self.prompt},
        )

    def init_all(self):
        self.init_embedding_model()
        self.load_FAISS_DB_retriever()
        self.init_tokenizer()
        self.init_model()
        self.init_pipeline()
        self.init_llm()
        self.init_prompt_template()
        self.init_qa_chain()

chatbot = MedicalChatbot(llm_model_name="microsoft/phi-2", faiss_indices_path="faiss_indices")
bar.progress(20)
progress_status.write(str(20)+"%")

chatbot.init_embedding_model()
bar.progress(30)
progress_status.write(str(30)+"%")

chatbot.load_FAISS_DB_retriever()
chatbot.init_tokenizer()
bar.progress(40)
progress_status.write(str(40)+"%")

chatbot.init_model()
bar.progress(60)
progress_status.write(str(60)+"%")

chatbot.init_pipeline()
bar.progress(70)
progress_status.write(str(70)+"%")

chatbot.init_llm()
bar.progress(80)
progress_status.write(str(80)+"%")

chatbot.init_prompt_template()
bar.progress(90)
progress_status.write(str(90)+"%")

chatbot.init_qa_chain()
bar.progress(100)
progress_status.write(str(100)+"%")



# Get user input
user_input = st.text_input("Ask a medical question:")

# Generate response on button click
if st.button("Get Answer"):
    if user_input:
        # Generate response
        response = chatbot.generate_response(user_input)
        st.markdown(f"**Answer:** {response}")