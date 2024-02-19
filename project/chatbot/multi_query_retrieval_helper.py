
from typing import List
from langchain.chains import LLMChain
from pydantic import BaseModel, Field
from langchain.prompts import PromptTemplate
from langchain.output_parsers import PydanticOutputParser
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.chains import TransformChain
from langchain.chains import SequentialChain

# Output parser will split the LLM result into a list of queries
class LineList(BaseModel):
    # "lines" is the key (attribute name) of the parsed output
    lines: List[str] = Field(description="Lines of text")


class LineListOutputParser(PydanticOutputParser):
    def __init__(self) -> None:
        super().__init__(pydantic_object=LineList)

    def parse(self, text: str) -> LineList:
        lines = text.strip().split("\n")
        return LineList(lines=lines)

def get_multi_query_prompt_template():
    template = """
    Your task is to generate 3 different search queries that aim to
    answer the user question from multiple perspectives. The user questions
    are focused on Large Language Models, Machine Learning, and related
    disciplines.
    Each query MUST tackle the question from a different viewpoint, we
    want to get a variety of RELEVANT search results.
    Provide these alternative questions separated by newlines.
    Original question: {question}
    """

    multi_query_prompt = PromptTemplate(
        input_variables=["question"],
        template=template,
    )

    return multi_query_prompt

def get_multi_query_llm_chain(chatbot):
    multi_query_llm_chain = LLMChain(llm=chatbot.llm, prompt=chatbot.mq_prompt_template, output_parser=chatbot.mq_output_parser)
    return multi_query_llm_chain

def get_multi_query_retriever(chatbot):
    multi_query_retriever = MultiQueryRetriever(
        retriever=chatbot.retriever, llm_chain=chatbot.mq_llm_chain, parser_key="lines"
    )  # "lines" is the key (attribute name) of the parsed output
    return multi_query_retriever




def get_multi_query_retrieval_chain(chatbot):
    def multi_query_retrieval_transform(inputs: dict) -> dict:
        docs = chatbot.mq_retriever.get_relevant_documents(query=inputs["question"])
        docs = [d.page_content for d in docs]
        docs_dict = {
            "query": inputs["question"],
            "contexts": "\n---\n".join(docs)
        }
        return docs_dict

    multi_query_retrieval_chain = TransformChain(
    input_variables=["question"],
    output_variables=["query", "contexts"],
    transform=multi_query_retrieval_transform
    )
    return multi_query_retrieval_chain

def get_multi_query_qa_chain(chatbot):
    QA_PROMPT = PromptTemplate(
    input_variables=["query", "contexts"],
    template="""You are a helpful assistant who answers user queries using the
    contexts provided. If the question cannot be answered using the information
    provided say "I don't know".

    Contexts:
    {contexts}

    Question: {query}""",
    )

    # Chain
    multi_query_qa_chain = LLMChain(llm=chatbot.llm, prompt=QA_PROMPT)
    return multi_query_qa_chain


def get_multi_query_rag_chain(chatbot):
    rag_chain = SequentialChain(
      chains=[chatbot.mq_retrieval_chain, chatbot.mq_qa_chain],
      input_variables=["question"],  # we need to name differently to output "query"
      output_variables=["query", "contexts", "text"]
      )
    return rag_chain


def show_multi_query_logs():
    # Set logging for the queries
    import logging

    logging.basicConfig()
    logging.getLogger("langchain.retrievers.multi_query").setLevel(logging.INFO)


