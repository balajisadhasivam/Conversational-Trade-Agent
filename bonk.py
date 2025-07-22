import streamlit as st
from agent.persona import TraderPersona
from agent.chat_agent import build_rag_agent
from agent.data_utils import load_wallet_transactions, load_wallet_portfolio, load_token_feed
from langchain.schema import Document
import os
import logging
import re
import pandas as pd
import dotenv
dotenv.load_dotenv()

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logging.info('Streamlit app started.')

st.set_page_config(page_title="Conversational Trade Agent", page_icon="💹")
st.title("💹 Conversational Trade Agent")

# Paths
data_path = "data"
tx_path = os.path.join(data_path, "wallet_transaction_sample.csv")
portfolio_path = os.path.join(data_path, "wallet_portfolio_sample.csv")
feed_path = os.path.join(data_path, "token_feed_sample.csv")

# Load real data
tx_df = load_wallet_transactions(tx_path)
portfolio_df = load_wallet_portfolio(portfolio_path)
feed_df = load_token_feed(feed_path)

# Build dummy persona 
persona = TraderPersona(
    name="Alex",
    style="Real-Time Onchain Analyst",
    risk="Medium",
    experience="Intermediate",
    strategy_summary="I analyze real-time wallet and token data to make decisions.",
    preferences="Focus on top tokens and monitor transaction patterns."
)

# Build RAG agent
if 'qa' not in st.session_state:
    st.session_state['qa'] = build_rag_agent(data_path, persona)
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

# Text cleaner
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

# Form
with st.form(key="qa_form", clear_on_submit=True):
    user_input = st.text_input("Your question", key="user_input")
    submit = st.form_submit_button("Ask")
    if submit and user_input.strip():
        logging.info(f"User question: {user_input}")
        with st.spinner("Thinking..."):
            result = qa.invoke({"query": user_input})
            answer_text = result.get("answer") if isinstance(result, dict) else str(result)
            logging.info(f"LLM OUTPUT: {answer_text}")
            cleaned_answer = answer_text.strip()
            st.session_state['chat_history'].append({
                "question": user_input,
                "answer": cleaned_answer
            })

# Display chat
for chat in st.session_state['chat_history']:
    st.markdown(f"**You:** {chat['question']}")
    st.markdown(f"**Trader:** {chat['answer']}")




