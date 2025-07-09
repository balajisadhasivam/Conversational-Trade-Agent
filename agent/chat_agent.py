import os
import logging
from langchain_community.document_loaders import CSVLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.memory import ConversationSummaryBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import ChatPromptTemplate
from langchain_together import Together
from agent.persona import TraderPersona

# Constants
MODEL_NAME = 'meta-llama/Llama-3.3-70B-Instruct-Turbo'
EMBEDDING_MODEL = 'sentence-transformers/all-MiniLM-L6-v2'

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

def build_rag_agent(csv_path: str, persona: TraderPersona):
    # Load CSV trades as documents
    loader = CSVLoader(file_path=csv_path)
    docs = loader.load()

    # Embed and index trades
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    db = FAISS.from_documents(docs, embeddings)

    # Set retriever with k=5 for more variety
    retriever = db.as_retriever(search_kwargs={"k": 5})

    # LLM setup with reduced max_tokens to avoid verbosity
    llm = Together(model=MODEL_NAME, max_tokens=300)

    
    '''memory = ConversationSummaryBufferMemory(
        llm=llm,
        memory_key="chat_history",
        return_messages=False,
        output_key="answer"
    )'''
    memory=None

    # Prompt Template
    prompt_template = ChatPromptTemplate.from_template("""\
You are a professional trading agent named {trader_name}.

Only respond to questions about trading, your trades, or your strategy. If a question is unrelated (e.g., weather, politics, jokes), respond with:
"I'm trained to answer questions about trading and my trading strategy. I'm not able to answer such questions."

Use the following trader persona to guide your answers only when relevant:
- Style: {style}
- Risk Appetite: {risk}
- Experience: {experience}
- Strategy Summary: {strategy_summary}
- Preferences: {preferences}

Context:
{context}

Answer the user's question below as a single, concise paragraph.
Do not include any notes, explanations, formatting comments, or justification.
Respond only with the final answer, without repeating the question or stating that you are answering it.

User Question: {question}

Trader's Answer:
""")

    qa = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        combine_docs_chain_kwargs={"prompt": prompt_template.partial(
            trader_name=persona.name,
            style=persona.style,
            risk=persona.risk,
            experience=persona.experience,
            strategy_summary=persona.strategy_summary,
            preferences=persona.preferences
        )},
        return_source_documents=True,
        output_key="answer"
    )

    logging.info(f"RAG agent built with persona: {persona.describe()}")
    return qa
