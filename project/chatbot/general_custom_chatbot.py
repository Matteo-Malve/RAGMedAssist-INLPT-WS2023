import random
import torch
import os
import pandas as pd
import time
import pinecone
from IPython.display import Markdown, display
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter, TokenTextSplitter, CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from transformers import AutoConfig, BitsAndBytesConfig, AutoModelForCausalLM, AutoTokenizer, GenerationConfig, pipeline
from langchain import HuggingFacePipeline
from langchain.chains import RetrievalQA
from langchain import hub
from langchain.vectorstores import Pinecone
from langchain.prompts import PromptTemplate
from langchain_community.llms import GPT4All
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())
HF_AUTH="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"

import warnings
warnings.filterwarnings('ignore')

def import_necessary_modules():
    import random
    import torch
    import os
    import pandas as pd
    import time
    import pinecone
    from IPython.display import Markdown, display
    from transformers import AutoConfig, AutoTokenizer, AutoModel
    from sentence_transformers import SentenceTransformer
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from langchain.text_splitter import RecursiveCharacterTextSplitter, TokenTextSplitter, CharacterTextSplitter
    from langchain_community.vectorstores import FAISS
    from transformers import BitsAndBytesConfig, AutoModelForCausalLM, AutoTokenizer, GenerationConfig, pipeline
    from langchain import HuggingFacePipeline
    from langchain.chains import RetrievalQA
    from langchain import hub
    from langchain.vectorstores import Pinecone
    from langchain.prompts import PromptTemplate
    from langchain_community.llms import GPT4All
    from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

import warnings
warnings.filterwarnings('ignore')
    
class MedicalChatbot:
    def __init__(self, 
                 embedding_model_name='thenlper/gte-base',
                 llm_model_name="mistralai/Mistral-7B-Instruct-v0.1", 
                 retriever_name="pinecone",
                 init_all=False):
        self.embedding_model_name = embedding_model_name
        self.llm_model_name = llm_model_name
        self.retriever_name=retriever_name
        self.faiss_indices_path = "../data/faiss_indices"
        self.device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_built() else "cpu"
        if init_all:
            self.init_all()
        self.chat_history = []

    def init_embedding_model(self):
        model_kwargs = {'device': self.device}
        encode_kwargs = {'normalize_embeddings': True}
        self.embed_model = HuggingFaceEmbeddings(model_name=self.embedding_model_name, model_kwargs=model_kwargs,
                                            encode_kwargs=encode_kwargs)

    def load_db_retriever(self):
        
        if self.retriever_name == "pinecone":
           self.load_Pinecone_db_retriever() 
        elif self.retriever_name == "faiss":
            self.load_FAISS_db_retriever()
        else:
            raise ValueError("Invalid retriever name. Please use either 'pinecone' or 'faiss'")
            
    def load_FAISS_db_retriever(self):
        self.db = FAISS.load_local(f"{self.faiss_indices_path}/{self.embedding_model_name}", self.embed_model)
        self.retriever = self.db.as_retriever(search_type="similarity", search_kwargs={"k": 3})


    def load_Pinecone_db_retriever(self, index_name="pubmed", search_type="similarity", k=3):
        os.environ["PINECONE_API_KEY"]="1218c885-67e3-492f-b1ab-215405569e97"
        pc = pinecone.Pinecone()
        index = pc.Index(index_name)
        self.db = Pinecone(index, self.embed_model, 'text')
        self.retriever = self.db.as_retriever(search_type=search_type, search_kwargs={"k": k})


    def get_relevant_documents_for_query(self, query):
        return self.retriever.invoke(query)

    def load_local_gpt4all_model(self):
        self.llm = GPT4All(
            model=self.llm_model_name,
            max_tokens=2048,
        )

    def init_model(self):
        if self.device == "cuda":
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

        else:
            self.model = AutoModelForCausalLM.from_pretrained(self.llm_model_name, torch_dtype="auto",  trust_remote_code=True)


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
            device=None if self.device=="cuda" else self.device ,
            generation_config=generation_config,
            framework="pt",
    )

    def init_llm(self):
        self.llm = HuggingFacePipeline(
            pipeline=self.pipeline,
    )


    def init_prompt_template(self):

        prompt_template = \
        """
        Context information is below.
        {context}
        Given the context information and not prior knowledge, answer the query.
        Query: {question}
        Use maximum three sentences.
        Answer:
        """
        self.prompt=PromptTemplate.from_template(prompt_template)
        #self.prompt  = hub.pull("rlm/rag-prompt", api_url="https://api.hub.langchain.com")

    def init_qa_chain(self):
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.retriever,
            return_source_documents=True,
            chain_type_kwargs={"prompt": self.prompt},
        )

    def init_all(self):
        self.init_embedding_model()
        self.load_db_retriever()
        if "GPT4ALL" in self.llm_model_name.upper():
            self.load_local_gpt4all_model()
        else:
            self.init_tokenizer()
            self.init_model()
            self.init_pipeline()
            self.init_llm()
        self.init_prompt_template()
        self.init_qa_chain()

    def retrieve_doi_urls(self,response):
        response_documents = response['source_documents']
        base_url = "https://doi.org/"
        doi_urls=[]
        for document in response_documents:
            doi = document.metadata.get('DOI')
            if doi:
                doi_urls.append(base_url + doi)
        return doi_urls
                

    def generate_response(self, user_query):
        response = self.qa_chain({"query": user_query})
        self.chat_history.append(response)
        return response
    
    def answer_the_user(self, user_query):
        response = self.generate_response(user_query)
        display(Markdown(f"<span style='color:blue'><b>{response['query']}</b></span>"))
        display(Markdown(f"<p>{response['result']}</p>"))
        doi_urls = self.retrieve_doi_urls(response)
        if doi_urls:
            display(Markdown(f"<p>For more information, please refer to the following links:</p>"))
            for url in doi_urls:
                display(Markdown(f"<p><a href='{url}' target='_blank'>{url}</a></p>"))


    

    def clean_chat_history(self):
        self.chat_history = []

    def set_prompt_template(self, prompt_template, print_template=True):
        self.prompt = prompt_template
        self.init_qa_chain()
        display(Markdown("New prompt template set successfully."))
        if print_template:
            display(Markdown(f"New prompt template reads: <p>{self.prompt}</p>"))

    def generate_response_and_measure_time(self,query, display_sources=False):

        start_time = time.time()  # Start timing
        result = self.generate_response(query)
        execution_time = time.time() - start_time  # Calculate execution time

        # Print the execution time rounded to 2 decimal places
        display(Markdown(f"The execution time on {self.device} using {self.llm_model_name} is {round(execution_time, 2)} seconds.\n"))

        # Display the query and result
        display(Markdown(f"<span style='color:blue'><b>{result['query']}</b></span>"))
        display(Markdown(f"<span style='color:blue'><p>{result['result']}</p></span>"))
        if display_sources:
            display(Markdown(f"<p>{result['source_documents']}</p>"))

    def test_prompts(self,promts_list,query,display_sources=False):
        self.init_prompt_template()
        display(Markdown(f"<span style='color:red'>Default template:</span>"))
        display(Markdown(f"<p>{self.prompt}</p>"))
        self.generate_response_and_measure_time(query)
        for prompt in promts_list:
            tag=prompt.get("tag")
            display(Markdown(f"<span style='color:red'><b>Testing prompt: {tag}</b></span>"))
            self.set_prompt_template(prompt["prompt"], print_template=False)
            self.generate_response_and_measure_time(query)
