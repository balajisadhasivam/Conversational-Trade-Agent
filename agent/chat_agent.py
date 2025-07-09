import os
import logging
from langchain_community.document_loaders import CSVLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import ChatPromptTemplate
from langchain_together import Together
from agent.persona import TraderPersona

# Constants
MODEL_NAME = 'meta-llama/Llama-3.3-70B-Instruct-Turbo'
EMBEDDING_MODEL = 'sentence-transformers/all-MiniLM-L6-v2'

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

def build_rag_agent(csv_path: str, persona: TraderPersona):
    # Load trades
    loader = CSVLoader(file_path=csv_path)
    docs = loader.load()

    # Embedding & indexing
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    db = FAISS.from_documents(docs, embeddings)

    # Retriever with k=1 to reduce hallucinations
    retriever = db.as_retriever(search_kwargs={"k": 1})

    # LLM setup
    llm = Together(model=MODEL_NAME, max_tokens=300)

    # Improved Prompt Template
    prompt_template = ChatPromptTemplate.from_template("""\
You are a professional trading agent named {trader_name}.

Only respond to questions about trading, your trades, or your strategy. If a question is unrelated, respond:
"I'm trained to answer questions about trading and my trading strategy."

Use the following persona traits if relevant:
- Style: {style}
- Risk Appetite: {risk}
- Experience: {experience}
- Strategy: {strategy_summary}
- Preferences: {preferences}

You may refer to a relevant example from the context below only if it directly supports the question.

Context:
{context}

Respond ONLY to the question below. Do not answer any previous or implied questions. Write a single concise paragraph.

User Question: {question}

Trader's Answer:
""")

    # Fill in persona
    prompt = prompt_template.partial(
        trader_name=persona.name,
        style=persona.style,
        risk=persona.risk,
        experience=persona.experience,
        strategy_summary=persona.strategy_summary,
        preferences=persona.preferences
    )

    # Create QA chain with LLM and prompt
    qa_chain = load_qa_chain(llm=llm, chain_type="stuff", prompt=prompt)

    # Assemble RetrievalQA chain
    qa = RetrievalQA(
        retriever=retriever,
        combine_documents_chain=qa_chain,
        return_source_documents=True,
        output_key="answer"
    )

    logging.info(f"RAG agent built with persona: {persona.describe()}")
    return qa
