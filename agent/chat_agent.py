import os
import logging
import pandas as pd
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import ChatPromptTemplate
from langchain.schema import Document
from langchain_together import Together
from agent.persona import TraderPersona

# Constants
MODEL_NAME = 'mistralai/Mixtral-8x7B-Instruct-v0.1'
EMBEDDING_MODEL = 'sentence-transformers/all-MiniLM-L6-v2'

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

def build_rag_agent(csv_path: str, persona: TraderPersona):
    # Load CSV as dataframe
    df = pd.read_csv(csv_path)

    # Convert trade rows into natural-language descriptions
    def row_to_doc(row):
        return Document(
            page_content=(
                f"TRADE: {row['Asset']} | {row['Date']} | {row['Type']} | Price: {row['Price']} | P/L: {row['Profit/Loss']}\n"
                f"Trade ID: {row['Trade ID']}. "
                f"Type: {row['Type']}. "
                f"Asset: {row['Asset']}. "
                f"Date: {row['Date']}. "
                f"Price: {row['Price']}. "
                f"Holding Period: {row['Holding Period']} days. "
                f"Profit/Loss: {row['Profit/Loss']}. "
                f"Sentiment: {row['Sentiment']}. "
                f"Source: {row['Source']}. "
                f"This trade was a {row['Type']} of {row['Asset']} on {row['Date']} at a price of {row['Price']}. "
                f"Why did I trade {row['Asset']} on {row['Date']}? [This is a record of the trade.] "
                f"What was the reason for trading {row['Asset']} on {row['Date']}? [See trade details above.] "
            )
        )

    docs = [row_to_doc(row) for _, row in df.iterrows()]

    # Build FAISS vector index
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    faiss_dir = "faiss_index"
    if os.path.exists(faiss_dir):
        db = FAISS.load_local(faiss_dir, embeddings, allow_dangerous_deserialization=True)
        logging.info("Loaded FAISS index from disk.")
    else:
        db = FAISS.from_documents(docs, embeddings)
        db.save_local(faiss_dir)
        logging.info("Built and saved FAISS index to disk.")

    # Use retriever with slightly higher k for better match rate
    retriever = db.as_retriever(search_kwargs={"k": 8})

    # LLM setup
    llm = Together(model=MODEL_NAME, max_tokens=300)

    # Persona-based prompt
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

If the user's question asks about a specific trade (for a particular asset and date):
- Scan the context for a line starting with 'TRADE: [ASSET] | [DATE]'.
    - If you find such a line, you MUST use the information from that line and the following details to answer the user's question. You MUST NOT use the fallback response if the trade is present in the context. Ignore unrelated context.
    - If you do NOT find such a line, you MUST respond exactly: "I did not make a trade for [ASSET] on [DATE]." (Replace [ASSET] and [DATE] with the asset and date from the user's question.)
For all other questions unrelated to trading, respond: "I'm trained to answer questions about trading and my trading strategy."

Respond ONLY to the question below. Do not answer any previous or implied questions. Write a single concise paragraph.

User Question: {question}

Trader's Answer:
""")

    prompt = prompt_template.partial(
        trader_name=persona.name,
        style=persona.style,
        risk=persona.risk,
        experience=persona.experience,
        strategy_summary=persona.strategy_summary,
        preferences=persona.preferences
    )

    # Create RetrievalQA chain
    qa_chain = load_qa_chain(llm=llm, chain_type="stuff", prompt=prompt)

    qa = RetrievalQA(
        retriever=retriever,
        combine_documents_chain=qa_chain,
        return_source_documents=True,
        output_key="answer"
    )

    logging.info(f"RAG agent built with persona: {persona.describe()}")
    return qa
