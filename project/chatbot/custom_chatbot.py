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
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.retrievers import BM25Retriever, EnsembleRetriever
from langchain.text_splitter import (
    CharacterTextSplitter,
    RecursiveCharacterTextSplitter,
    TokenTextSplitter,
)
#from langchain.vectorstores import Pinecone
from langchain_community.vectorstores import Pinecone
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate

# Suppress warnings
warnings.filterwarnings("ignore")

print(torch.cuda.is_available()) 

# Automatically find and load the .env file
load_dotenv(find_dotenv())


class MedicalChatbot:
    def __init__(self, cfg):
        self.device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_built() else "cpu"
        )
        self.chat_history = []
        self.cfg = cfg
        self.qa_chain = self.init_qa_chain()

    def generate_response(self, user_query):
        response = self.qa_chain({"query": user_query})
        self.chat_history.append(response)
        return response

    def init_embedding_model(self):
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
            return self.load_db_retriever(name)
        elif name == "bm25":
            return self.load_bm25_retriever()

    def load_db_retriever(self, retriever_name):
        if retriever_name == "pinecone":
            return self.load_pinecone_db_retriever()
        else:
            return self.load_faiss_db_retriever()

    def load_faiss_db_retriever(self):
        faiss_index_path = f"{self.cfg['retrievers']['faiss']['faiss_index_path']}{self.cfg['embedding_model']}"
        db = FAISS.load_local(faiss_index_path, self.init_embedding_model())
        return db.as_retriever(
            search_type="similarity",
            search_kwargs={"k": self.cfg["retrievers"]["faiss"]["topk"]},
        )

    def load_pinecone_db_retriever(self):
        pinecone_api_key = os.getenv("PINECONE_API_KEY")
        if not pinecone_api_key:
            raise ValueError(
                "Pinecone API key not found. Please set it in the .env file."
            )
        pc = pinecone.init(api_key=pinecone_api_key)
        index = pc.Index(self.cfg["retrievers"]["pinecone"]["index_name"])
        db = Pinecone(index, self.embed_model, "text")
        return db.as_retriever(
            search_type=self.cfg["retrievers"]["pinecone"]["search_type"],
            search_kwargs={"k": self.cfg["retrievers"]["pinecone"]["topk"]},
        )

    def load_ensemble_retriever(self):
        ensemble_list = []
        weights = self.cfg["ensemble"]["weights"]
        for name, value in self.cfg["retrievers"].items():
            retriever = self.load_retrievers(name)
            ensemble_list.append(retriever)

        if not ensemble_list:
            raise ValueError("No valid retrievers were loaded.")
        return EnsembleRetriever(
            retrievers=ensemble_list, weights=weights
        )

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
            #torch_dtype=torch.float32 if self.device.type == "cuda" else "auto",
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
        generation_config.max_new_tokens=512
        generation_config.temperature=self.cfg["llm_model"]["temperature"]
        generation_config.top_p=self.cfg["llm_model"]["top_p"]
        generation_config.do_sample=True
        generation_config.repetition_penalty=self.cfg["llm_model"]["repetition_penalty"]
    
        pipe = pipeline(
            "text-generation",
            model=self.init_model(),
            tokenizer=self.init_tokenizer(),
            #device=None if self.device=="cuda" else self.device,
            generation_config=generation_config,
        )
        return HuggingFacePipeline(
            pipeline=pipe
        )

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
        template = """<s> [INST] You are a helpful assistant for biomedical question-answering tasks. 
                    Use only the following retrieved context to answer the question. If the answer is not in the context,
                    just say that you don't know. 
                    Provide a response strictly based on the information requested in the query.[/INST] </s> 
                    [INST] Question: {question} 
                    Context: {context} 
                    Answer: [/INST]"""
        return PromptTemplate.from_template(template)
        #return hub.pull("rlm/rag-prompt", api_url="https://api.hub.langchain.com")
        
    def clean_chat_history(self):
        self.chat_history = []



if __name__ == "__main__":

    with open("cfg.yaml", "r") as file:
        cfg = yaml.safe_load(file)

    # Define different testing queries
    qa_evalset_queries = [
        "Does tick-borne encephalitis carry a high risk of incomplete recovery in children?",
        "Is language dysfunction associated with age of onset of benign epilepsy with centrotemporal spikes in children?",
        "Is occupational outcome in bipolar disorder predicted by premorbid functioning and intelligence?",
        "Does the CACNA1C risk allele selectively impact on executive function in bipolar type I disorder?",
        "Does emotional intelligence predict breaking bad news skills in pediatric interns?",
        "Cognitive recovery after severe traumatic brain injury in children/adolescents and adults: similar positive outcome but different underlying pathways?",
        "Is bilateral hearing loss associated with decreased nonverbal intelligence in US children aged 6 to 16 years?",
        "Is the association between intelligence and lifespan mostly genetic?",
        "What are the effects of α1-antitrypsin (AAT) treatment on chronic fatigue syndrome (CFS) based on a case study involving a 49-year-old woman?",
        "Are cerebral white matter fractional anisotropy and tract volume as measured by MR imaging associated with impaired cognitive and motor function in pediatric posterior fossa tumor survivors?"
    ]
    chatgpt_queries = [
        "Can learning a second language improve cognitive skills and intelligence?",
        "Is there a link between early life stress and its long-term impact on cognitive development and intelligence?",
        "Can a healthy diet during childhood improve intelligence and academic performance?",
        "How do environmental factors during childhood affect intelligence outcomes in adulthood?",
        "How does sleep quality impact learning abilities and intelligence in students?"
    ]
    unrelated_topic_queries = [
        "What are your favorite comedies?",
        "Which football team do you think will win the world cup this year?"
    ]
    queries = [qa_evalset_queries, chatgpt_queries, unrelated_topic_queries]

    query_list_names = ["Queries from QA-Evaluationset", "Queries from ChatGPT-4", "Unrelated Topic Quries"]

    # Call chatbot
    chatbot = MedicalChatbot(cfg)

    # Initialize list to store query-response pairs
    query_response_pairs = []

    #Open a Markdown file for writing the results
    with open("hybrid_search_results_1_0.md", "w") as md_file:
        md_file.write(f"# Testing of Differnt Weights for Hybrid Search\n")
        md_file.write(f"**BM25 Keyword Search: {cfg['ensemble']['weights'][0]}, {cfg['embedding_model']} Vector Search: {cfg['ensemble']['weights'][1]}**\n")
        md_file.write(f"LLM parameters: temp={cfg['llm_model']['temperature']}, topp={cfg['llm_model']['top_p']}, rep_penalty={cfg['llm_model']['repetition_penalty']}\n\n")
        prompt = chatbot.get_prompt()
        md_file.write(f"## Custom Prompt Template:\n```python\n{prompt}\n```\n\n")

        for name, query_list in zip(query_list_names, queries):
            md_file.write(f"### {name}\n")
            for query in query_list:
                start_time = time.time()  # Start timing
                result = chatbot.generate_response(query)
                execution_time = time.time() - start_time  # Calculate execution time

                # Store query-response pair in list
                query_response_pairs.append([query, result['result']])

                # Write the query, execution time, and result to the Markdown file
                md_file.write(f"## Query:\n*{query}*\n\n")
                md_file.write(f"**Execution Time:**\n{round(execution_time, 2)} seconds on {chatbot.device} using {cfg['llm_model']['name']}.\n\n")
                md_file.write(f"### Response:\n{result['result']}\n\n")
                # Add a horizontal rule for separation between entries
                md_file.write("---\n\n")


    # Save query-response pairs in .txt file for easier extraction
    with open("query_response_pairs_hybrid_search_1_0.txt", "w") as txt_file:
        # Convert list of pairs to a string representation
        txt_content = str(query_response_pairs)
        txt_file.write(txt_content)

    print("Results have been saved.")



    

