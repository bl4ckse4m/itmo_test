import argparse

from langchain import hub
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langgraph.graph import START, StateGraph
from pydantic import BaseModel, Field
from typing_extensions import List, TypedDict

from langchain_core.output_parsers import PydanticOutputParser
import store
from config import OPENAI_API_KEY, TAVILY_API_KEY
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.retrievers import TavilySearchAPIRetriever
from langgraph.graph import END
# Define prompt for question-answering
prompt = hub.pull("rlm/rag-prompt")

embeddings = OpenAIEmbeddings(model="text-embedding-3-large")



tavily = TavilySearchAPIRetriever(k=3, api_key=TAVILY_API_KEY)

llm = ChatOpenAI(api_key=OPENAI_API_KEY, model="gpt-4o-mini")

vector_store = store.load()


# Define state for application
class State(TypedDict):
    question: str
    context: List[Document]
    answer: str


class Result(BaseModel):
    """Answer to user query."""

    answer: int | None = Field(..., description="""
    Numeric value representing the correct answer if question is a multiple choice question. 
    If question is not a multiple choice question, the value should be null.""")
    reasoning: str = Field(
        ..., description="""Contains explanation or additional information about the answer."""
    )
    dont_know: bool = Field(
        ..., description="""Indicates that context is not enough to answer the question or the model is not sure about the answer."""
    )


# Define application steps
def retrieve(state: State):
    retrieved_docs = vector_store.similarity_search(state["question"])
    return {"context": retrieved_docs}


def websearch(state:State):
   return {'context': tavily.invoke(state["question"])}

def generate(state: State):
    parser = PydanticOutputParser(pydantic_object=Result)

    docs_content = "\n\n".join(doc.page_content for doc in state["context"])

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
You are chat bot that should answer questions regarding ITMO university.
The user may ask two kinds of questions:
*  a questions with enumerated list of answer options that you should answer with the number
*  a generic question that you should answer with a short text in reasoning field  

Use the following context while answering the question:

{context}

{format_instructions}
""",
            ),
            ("human", "{question}"),
        ]
    ).partial(format_instructions=parser.get_format_instructions())



    messages = prompt.invoke({"question": state["question"], "context": docs_content})
    output = llm.invoke(messages)
    return {'answer' :  parser.invoke(output)}


def is_rag_dont_know(state: State) -> str:
    # Check if RAG result is good enough (replace with actual logic)
    return  'websearch' if state['answer'].dont_know else END

# Compile application and test
graph_builder = StateGraph(State)
graph_builder.add_node("retrieve", retrieve)
graph_builder.add_node("websearch", websearch)
graph_builder.add_node("generate", generate)
graph_builder.add_node("generate1", generate)


graph_builder.add_edge(START, "retrieve")
graph_builder.add_edge("retrieve", "generate")
graph_builder.add_conditional_edges(
    'generate', is_rag_dont_know
)
graph_builder.add_edge("websearch", "generate1")
graph_builder.add_edge("generate1", END)

graph = graph_builder.compile()

def answer(query: str):
    result = graph.invoke({"question": query})
    return result


if __name__ == "__main__":

    # Create an ArgumentParser object parser = argparse.ArgumentParser(description="Query processor")
    parser = argparse.ArgumentParser(description="Answer question about ITMO")
    # Add an argument for the query
    parser.add_argument("--query", type=str, help="Enter your query")

    # Parse the arguments
    args = parser.parse_args()

    # Print the query value
    result = answer(args.query)
    print(result["answer"])