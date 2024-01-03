from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI
from langchain.vectorstores import Pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
import pinecone
import os
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import AIMessage, HumanMessage

load_dotenv()

pinecone.init(
    api_key=os.getenv("PINECONE_API_KEY"),
    environment=os.getenv("PINECONE_ENV"),
)

index_name = 'penn-courses'
embeddings = OpenAIEmbeddings()
vectorstore = Pinecone.from_existing_index(index_name, embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 8})

llm = ChatOpenAI(model_name="gpt-4", temperature=0)


contextualize_system_prompt = """Given a chat history and the latest user question \
which might reference context in the chat history, formulate a standalone question \
which can be understood without the chat history. Do NOT answer the question, \
just reformulate it if needed and otherwise return it as is."""
contextualize_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_system_prompt),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}"),
    ]
)
contextualize_chain = contextualize_prompt | llm | StrOutputParser()


system_prompt = """You are an AI college advisor focused on advising Penn students on anything relating to courses. Some of your responsibilities include recommending courses and constructing a course plan. Remember to personlize your responses to the student's interests and strengths. If you don't know the answer, just say that you don't know, don't try to make up an answer. Be as detailed as possible in your response. Give detailed reasons to your advice. Introdce yourself and your functions to the student when prompted. You are given relevant context to answer students' questions as follows:

{context}"""

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}"),
    ]
)


def contextualized_question(input: dict):
    if input.get("chat_history"):
        return contextualize_chain
    else:
        return input["question"]


rag_chain = (
    RunnablePassthrough.assign(
        context=contextualized_question | retriever
    )
    | prompt
    | llm
)


chat_history = []


while True:
    question = input("Query: ")
    response = rag_chain.stream(
        {"question": question, "chat_history": chat_history})
    final_response = ""
    for chunk in response:
        final_response += chunk.content
        print(chunk.content, end="", flush=True)
    print("\n")
    chat_history.extend([HumanMessage(content=question),
                        AIMessage(content=final_response)])
