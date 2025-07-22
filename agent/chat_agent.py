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
from agent.data_utils import load_wallet_transactions, load_wallet_portfolio, load_token_feed

# Constants
MODEL_NAME = 'mistralai/Mixtral-8x7B-Instruct-v0.1'
EMBEDDING_MODEL = 'sentence-transformers/all-MiniLM-L6-v2'

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')


def build_transaction_docs(tx_df: pd.DataFrame, token_feed_df: pd.DataFrame) -> list:
    feed = token_feed_df[["token_address", "token_symbol", "token_name"]].drop_duplicates()
    tx_df = tx_df.merge(feed, left_on="token_address", right_on="token_address", how="left")

    docs = []
    for _, row in tx_df.iterrows():
        content = (
            f"TRANSACTION: {row.get('transaction_type', '')} "
            f"{row.get('from_token_transaction_value', '')} {row.get('token_symbol', 'UNKNOWN')} "
            f"on {row.get('block_time', '')} for a value of ${row.get('transaction_value', '')}. "
            f"From: {row.get('from_address', '')} To: {row.get('to_address', '')}. "
            f"TxHash: {row.get('tx_hash', '')} Chain: {row.get('chain_name', '')}."
        )
        docs.append(Document(page_content=content.strip()))
    return docs


def build_portfolio_docs(portfolio_df, token_feed_df):
    token_lookup = token_feed_df[["token_address", "token_symbol", "token_name"]].drop_duplicates()
    enriched = portfolio_df.merge(token_lookup, on="token_address", how="left")

    docs = []
    for _, row in enriched.iterrows():
        token_symbol = row.get('token_symbol') or 'UNKNOWN'
        token_name = row.get('token_name') or ''
        doc_str = (
            f"PORTFOLIO: Wallet {row.get('wallet_address', '')} on {row.get('chain_name', '')} holds "
            f"{row.get('available_token_balance', '')} {token_symbol} ({token_name}) "
            f"worth {row.get('available_token_balance_in_usd', '')} USD "
            f"at price {row.get('token_price_in_usd', '')} USD/token."
        )
        docs.append(Document(page_content=doc_str.strip()))
    return docs


def build_token_feed_docs(feed_df: pd.DataFrame) -> list:
    docs = []
    for _, row in feed_df.iterrows():
        content = (
            f"TOKEN FEED: {row.get('token_symbol', 'UNKNOWN')} ({row.get('token_name', '')}) is trading at "
            f"${row.get('current_price_in_usd', '')} with a market cap of ${row.get('current_market_cap', '')}. "
            f"Volume: ${row.get('current_volume', '')} | Liquidity: ${row.get('current_liquidity', '')}. "
            f"Price change (24h): {row.get('price_change_percent_24_hr', '')}%. "
            f"Last updated on {row.get('date', '')}."
        )
        docs.append(Document(page_content=content.strip()))
    return docs


def build_rag_agent(base_path: str, persona: TraderPersona):
    # Load data
    tx_df = load_wallet_transactions(os.path.join(base_path, "wallet_transaction_sample.csv"))
    portfolio_df = load_wallet_portfolio(os.path.join(base_path, "wallet_portfolio_sample.csv"))
    feed_df = load_token_feed(os.path.join(base_path, "token_feed_sample.csv"))

    tx_docs = build_transaction_docs(tx_df, feed_df)
    portfolio_docs = build_portfolio_docs(portfolio_df, feed_df)
    feed_docs = build_token_feed_docs(feed_df)

    all_docs = tx_docs + portfolio_docs + feed_docs

    # Vector embedding
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    faiss_dir = "faiss_index"
    if os.path.exists(faiss_dir):
        db = FAISS.load_local(faiss_dir, embeddings, allow_dangerous_deserialization=True)
        logging.info("Loaded FAISS index from disk.")
    else:
        db = FAISS.from_documents(all_docs, embeddings)
        db.save_local(faiss_dir)
        logging.info("Built and saved FAISS index to disk.")

    retriever = db.as_retriever(search_kwargs={"k": 8})

    # Setup 
    llm = Together(
        model=MODEL_NAME,
        max_tokens=1024,
        temperature=0.7,
        top_p=0.9
    )

    # Prompt template
    prompt_template = ChatPromptTemplate.from_template("""\
You are a professional crypto trading assistant named {trader_name}.

You are helping the user understand the purpose of trades, identify token exposures, and make decisions like rebalancing or profit-taking.

You have access to:
- Wallet transactions
- Token holdings
- Token market feed

---

💬 When answering questions:
- Use clear headings (e.g., "📊 Token Exposure Breakdown")
- Show ranked lists or markdown tables if relevant
- Provide a short and actionable 🧠 Recommendation at the end

---

Context:
{context}

User Question:
{question}

---

Your Response (follow this format):

📊 Token Exposure Breakdown  
<optional ranked list or table>

🧠 Recommendation  
<1-2 line actionable advice with reasoning>
""")


    prompt = prompt_template.partial(
        trader_name=persona.name
    )

    qa_chain = load_qa_chain(llm=llm, chain_type="stuff", prompt=prompt)

    qa = RetrievalQA(
        retriever=retriever,
        combine_documents_chain=qa_chain,
        return_source_documents=True,
        output_key="answer"
    )

    logging.info(f"RAG agent built using Together (Mixtral) and real data for persona: {persona.describe()}")
    return qa
