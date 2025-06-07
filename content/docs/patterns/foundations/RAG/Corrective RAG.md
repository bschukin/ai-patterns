# Corrective RAG (C-RAG)

<br>
{{% hint info %}}

Corrective RAG is an improvement on top of conventional RAG that can use LLM as a judge to refine RAG responses.

{{% /hint %}}

## Algorithm

1. The query is routed to the system, which is taken over by the vector store.
2. The vector store returns the document chunks fetched by the LLM and the LLM responds with a conversational summary referring to the document chunks as context.
3. Another LLM (or maybe the same one) fetches the response and tries to determine how much the LLM is hallucinating.
4. If the LLM is hallucinating, the LLM tries to regenerate the answer.
5. If the LLM is not hallucinating, the LLM tries to determine whether it is on point and resolves the user’s query.
6. If the answer does not resolve the user’s query, the system (an LLM) tries to regenerate <br>the query so that the vector store can retrieve more relevant chunks.

```python

"""
1. Ensure docker is installed and running (https://docs.docker.com/get-docker/)
2. pip install -qU langchain_postgres
3. Run the following command to start the postgres container:
   
docker run \
    --name pgvector-container \
    -e POSTGRES_USER=langchain \
    -e POSTGRES_PASSWORD=langchain \
    -e POSTGRES_DB=langchain \
    -p 6024:5432 \
    -d pgvector/pgvector:pg16
4. Use the connection string below for the postgres container

"""

import os
from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_postgres.vectorstores import PGVector
from pydantic import BaseModel, Field
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from typing import List
from typing_extensions import TypedDict
from langchain.schema import Document
from langgraph.graph import END, StateGraph, START

os.system("clear")

class GraphState(TypedDict):
    question: str
    generation: str
    documents: List[Document]

class GradeHallucinations(BaseModel):
  binary_score:str = Field(description="Answer is grounded in the facts, 'yes' or 'no'")

class GradeAnswers(BaseModel):
  binary_score:str = Field(description="Answer addresses the question, 'yes' or 'no'")

loader = PyPDFLoader("Activation_Functions.pdf")
pages = loader.load_and_split()

connection = "postgresql+psycopg://langchain:langchain@localhost:6024/langchain"

key = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpZCI6IjYwMjc0OTRlLWI1NTEtNGFjNy1iZTA0LTQ4NDEzMjdlZjZmMiIsImlzRGV2ZWxvcGVyIjp0cnVlLCJpYXQiOjE3NDQ0MzU0ODEsImV4cCI6MjA2MDAxMTQ4MX0.Tx3NvjUYdGBY3UskbrvFZnhfBxzFuDAp3b7Ii_BXGr8'
model = ChatOpenAI(model="gpt-3.5-turbo", base_url='https://bothub.chat/api/v2/openai/v1', api_key=key)

vectore_store = PGVector.from_documents(
    documents=pages, embedding=OpenAIEmbeddings(base_url='https://bothub.chat/api/v2/openai/v1', api_key=key), connection=connection)

retriever = vectore_store.as_retriever()

generate_template = """Answer the question based only on the
                    following context:
                    {context}
                    Question: {question}
                    """

generation_prompt = ChatPromptTemplate.from_template(generate_template)
output_parser = StrOutputParser()
generation_chain = generation_prompt | model | output_parser

#set up a LangChain pipeline to grade hallucinations from an LLM  response
llm_grader = model.with_structured_output(GradeHallucinations)

system = """You are a grader assessing whether an LLM
        generation is grounded in / supported by a set of retrieved
        facts. \n
     Give a binary score 'yes' or 'no'. 'Yes' means that the
     answer is grounded in / supported by the set of facts.
     """

hall_prompt = ChatPromptTemplate.from_messages([
    ("system",system),
    ("human","Set of facts: \n\n {documents} \n\n LLM generation: {generation}")
])

hallucination_grader = hall_prompt|llm_grader

llm_grader_ans = model.with_structured_output(GradeAnswers)

system = """You are a grader assessing whether an answer
    addresses / resolves a question \n
     Give a binary score 'yes' or 'no'. Yes' means that the
     answer resolves the question."""

answer_prompt = ChatPromptTemplate.from_messages([
    ("system",system),
    ("human", "User question: \n\n {question} \n\n LLMgeneration: {generation}")
])

answer_grader = answer_prompt|llm_grader_ans

#Finally, prepare a prompt and a chain to rewrite the user query.

rewrite_template = """You a question re-writer that converts an
        input question to a better version that is optimized \n
        for vectorstore retrieval. Look at the input and try to
        reason about the underlying semantic intent / meaning."""
rewrite_prompt = ChatPromptTemplate.from_template(rewrite_template)

output_parser_rewrite = StrOutputParser()
sr = RunnableParallel({"question":RunnablePassthrough()})

rewrite_chain = sr | rewrite_prompt | model | output_parser_rewrite

def retrieve(state):
  question = state["question"]
  documents = retriever.invoke(question)
  return {"documents":documents,"question":question}

def format_docs(docs):
  return "\n\n".join(doc.page_content for doc in docs)

def generate(state):
  question = state["question"]
  documents = state["documents"]
  generation = generation_chain.invoke({"context":format_docs(documents),"question":question})
  return {"documents": documents, "question": question,  "generation": generation}

def transform_query(state):
  question = state["question"]
  documents = state["documents"]
  better_question = rewrite_chain.invoke({"question":question})
  return {"documents": documents, "question": better_question}

def grade_generation_v_documents_and_question(state):
    question = state["question"]
    documents = state["documents"]
    generation = state["generation"]
    hg = hallucination_grader.invoke({"documents":documents,"generation":generation})
    if hg.binary_score=="yes":
       ag = answer_grader.invoke({"question":question,"generation":generation})
       if ag.binary_score=="yes":
        return "useful"
       else:
        return "not useful"
    else:
     return "not supported"


workflow = StateGraph(GraphState)
workflow.add_node("retrieve",retrieve)
workflow.add_node("generate",generate)
workflow.add_node("transform_query",transform_query)

workflow.add_edge(START,"retrieve")
workflow.add_edge("retrieve","generate")
workflow.add_conditional_edges("generate",grade_generation_v_documents_and_question,
                               {
                                   "not supported":"generate",
                                   "not useful":"transform_query",
                                   "useful":END})

app = workflow.compile()
inputs = {"question": "What is Activation Function?"}
for output in app.stream(inputs):
  print(output)

```