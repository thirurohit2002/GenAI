import os
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough #Runnable, helps to pass the input(in case if we have to add additional input while invoking) when it is invoked
from operator import itemgetter

load_dotenv()

print("Initializing components")

embeddings = OpenAIEmbeddings(openai_api_key=os.getenv("OPENROUTER_API_KEY"), base_url="https://openrouter.ai/api/v1")

llm = ChatOpenAI(openai_api_key=os.getenv("OPENROUTER_API_KEY"), base_url="https://openrouter.ai/api/v1")

vectorstore = PineconeVectorStore(index_name=os.getenv('INDEX_NAME'), embedding=embeddings)

retriever = vectorstore.as_retriever(search_kwargs={"k": 3}) #Return an object of type VectorStoreRetriever(this has a search function), i.e., vector store with searching capabilities
#retriever has only top 3 chunks/documents based on order of important or relevance for the given query

prompt_template = ChatPromptTemplate.from_template(
    """Answer the question based only on the following context:

    {context}

    Question: {question}

    Provide a detailed answer:""") #augmentation of the prompt

def format_docs(docs):
    """Format retrieved documents into a single string""" #We may get 3 chunks, so we are joining them into a single string
    return "\n\n".join(doc.page_content for doc in docs)

def retrieval_chain_without_lcel(query:str):
    """
    Simple retrieval chain without LCEL.
    Manually retrieves documents, formats them, and then generates response
    """
    docs = retriever.invoke(query) #runs _aget_relevant_documents method through the pinecone sdk, and gives us 3 chunks based on the retriever config
    context = format_docs(docs)

    messages = prompt_template.format_messages(context=context, question=query)

    response = llm.invoke(messages)

    return response.content

def retrieval_chain_with_lcel(): #no args -> because this method is going to return a langchain chain, that chain is going to be a langchain runnable, we can go and use invoke on top of that.
    #With Lcel we get more observability
    """
    Retrieval chain with LCEL.
    Return a chain that can be invoked with {"question": "......"}
    """
    retrival_chain = RunnablePassthrough.assign( #input_dict = {"question": "What is pinecone in machine learning?"}
        context = itemgetter('question') | retriever | format_docs #Result: {"question": "What is pinecone in machine learning?", "context": "doc1\n\ndoc2\n\ndoc3"} -> this will be passed through prompt_template
    ) | prompt_template | llm | StrOutputParser()
    return retrival_chain


if __name__ == "__main__":
    print("Retrieving...")

    query = "What is Pinecone in machine learning"

    # result_without_lcel = retrieval_chain_without_lcel(query)
    #
    # print(result_without_lcel)

    chain_with_lcel = retrieval_chain_with_lcel()

    result_with_lcel = chain_with_lcel.invoke({"question": query})

    print(result_with_lcel)

