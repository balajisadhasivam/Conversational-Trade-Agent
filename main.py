import streamlit as st
from agent.persona import assign_persona_from_trades
from agent.chat_agent import build_rag_agent
from langchain.schema import Document
from agent.data_utils import load_trades, generate_synthetic_trades
import os
from dotenv import load_dotenv
import logging
import re
import pandas as pd

load_dotenv()

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logging.info('Streamlit app started.')

csv_path = os.path.join('data', 'trades.csv')
if not os.path.exists(csv_path):
    os.makedirs('data', exist_ok=True)
    generate_synthetic_trades(num_trades=500, out_path=csv_path)

st.set_page_config(page_title="Conversational Trade Agent", page_icon="💹")
st.title("💹 Conversational Trade Agent")

# Load trades and assign persona
trades = load_trades(csv_path)
persona = assign_persona_from_trades(trades)

# Build agent
if 'qa' not in st.session_state:
    st.session_state['qa'] = build_rag_agent(csv_path, persona)
qa = st.session_state['qa']

with st.expander("Trader Persona Details", expanded=True):
    st.markdown(f"""
    **Trader Name:** {persona.name}  
    **Style:** {persona.style}  
    **Risk Appetite:** {persona.risk}  
    **Experience Level:** {persona.experience}  
    **Strategy:** {persona.strategy_summary}  
    **Preferences:** {persona.preferences}
    """)

st.markdown("---")
st.markdown("#### Ask the trader agent a question:")

# Initialize chat history
if 'chat_history' not in st.session_state:
    st.session_state['chat_history'] = []

# Clear chat
if st.button("Clear Chat"):
    st.session_state['chat_history'] = []

# Text cleaning functions
def clean_response_to_single_paragraph(text: str) -> str:
    text = re.sub(r'\n+', ' ', text.strip())
    sentences = re.split(r'(?<=[.!?])\s+', text)
    seen = set()
    result = []
    for sent in sentences:
        stripped = sent.strip()
        if stripped and stripped not in seen:
            result.append(stripped)
            seen.add(stripped)
    return ' '.join(result)

def strip_qa_leaks(text: str) -> str:
    return re.sub(r"(User Question:|Trader's Answer:).*", "", text, flags=re.IGNORECASE | re.DOTALL).strip()

# Remove fallback phrase if appended after a valid answer
FALLBACK_PHRASE = "I'm trained to answer questions about trading and my trading strategy."
def strip_fallback_phrase(text: str) -> str:
    # Remove the fallback phrase if it appears at the end or after a valid answer
    return re.sub(rf"\s*{re.escape(FALLBACK_PHRASE)}\s*", "", text).strip()

# Helper to extract asset and date from user question
ASSET_DATE_PATTERN = re.compile(r"(?:buy|sell|trade|close|reasoning for|reason for|reasoning|reason)?\s*(?P<asset>[A-Z]{2,10})\s*(?:trade|position|on)?\s*(?P<date>20\d{2}-\d{2}-\d{2})", re.IGNORECASE)
def extract_asset_date(question):
    match = ASSET_DATE_PATTERN.search(question)
    if match:
        return match.group('asset').upper(), match.group('date')
    return None, None

# Helper to build trade doc in same format as in chat_agent.py
TRADE_DOC_TEMPLATE = ("TRADE: {Asset} | {Date} | {Type} | Price: {Price} | P/L: {Profit_Loss}\n"
                     "Trade ID: {Trade_ID}. Type: {Type}. Asset: {Asset}. Date: {Date}. Price: {Price}. "
                     "Holding Period: {Holding_Period} days. Profit/Loss: {Profit_Loss}. Sentiment: {Sentiment}. Source: {Source}. "
                     "This trade was a {Type} of {Asset} on {Date} at a price of {Price}. "
                     "Why did I trade {Asset} on {Date}? [This is a record of the trade.] "
                     "What was the reason for trading {Asset} on {Date}? [See trade details above.] ")

def build_trade_doc(row):
    return TRADE_DOC_TEMPLATE.format(
        Asset=row['Asset'], Date=row['Date'], Type=row['Type'], Price=row['Price'],
        Profit_Loss=row['Profit/Loss'], Trade_ID=row['Trade ID'],
        Holding_Period=row['Holding Period'], Sentiment=row['Sentiment'], Source=row['Source']
    )

# Load full trades DataFrame for post-retrieval filter
trades_df = pd.read_csv(csv_path)

# Form
with st.form(key="qa_form", clear_on_submit=True):
    user_input = st.text_input("Your question", key="user_input")
    submit = st.form_submit_button("Ask")
    if submit and user_input.strip():
        logging.info(f"User question: {user_input}")
        with st.spinner("Thinking..."):
            asset, date = extract_asset_date(user_input)
            # Efficient approach: if asset+date match exists, use only that trade as context
            trade_answered = False
            if asset and date:
                match = trades_df[(trades_df['Asset'] == asset) & (trades_df['Date'] == date)]
                if not match.empty:
                    trade_doc = build_trade_doc(match.iloc[0])
                    doc_obj = Document(page_content=trade_doc)
                    qa_chain = qa.combine_documents_chain
                    # LOGGING: Show the exact context and question sent to the LLM
                    logging.info(f"LLM INPUT (Efficient):\nContext:\n{trade_doc}\nQuestion: {user_input}")
                    result = qa_chain({
                        "input_documents": [doc_obj],
                        "question": user_input
                    })
                    logging.info(f"LLM OUTPUT (Efficient): {result['answer']}")
                    trade_answered = True
            if not trade_answered:
                # Fallback to normal retrieval
                # LOGGING: Show the question and the context docs retrieved
                retrieval_result = qa.retriever.get_relevant_documents(user_input)
                logging.info(f"LLM INPUT (Retrieval):\nContext:\n" + '\n---\n'.join([doc.page_content for doc in retrieval_result]) + f"\nQuestion: {user_input}")
                result = qa.invoke({"query": user_input})
                logging.info(f"LLM OUTPUT (Retrieval): {result['answer']}")
            logging.info(f"Raw agent answer: {result['answer']}")
            cleaned_answer = result['answer']
            st.session_state['chat_history'].append({
                "question": user_input,
                "answer": cleaned_answer
            })

# Chat history display
for chat in st.session_state['chat_history']:
    st.markdown(f"**You:** {chat['question']}")
    st.markdown(f"**Trader:** {chat['answer']}")
