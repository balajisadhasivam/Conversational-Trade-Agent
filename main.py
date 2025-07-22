import streamlit as st
from streamlit_chat import message
from agent.persona import TraderPersona
from agent.chat_agent import build_rag_agent
from agent.data_utils import load_wallet_transactions, load_wallet_portfolio, load_token_feed
import os
import logging
import pandas as pd
import dotenv
dotenv.load_dotenv()

# Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logging.info('Streamlit app started.')

# Streamlit UI Config
st.set_page_config(page_title="Conversational Trade Agent", page_icon="💹", layout="wide")
st.title("💹 Conversational Trade Agent")
if st.button("Clear Chat"):
    st.session_state['chat_history'] = []

# Paths
data_path = "data"
tx_path = os.path.join(data_path, "wallet_transaction_sample.csv")
portfolio_path = os.path.join(data_path, "wallet_portfolio_sample.csv")
feed_path = os.path.join(data_path, "token_feed_sample.csv")

# Load Data
tx_df = load_wallet_transactions(tx_path)
portfolio_df = load_wallet_portfolio(portfolio_path)
feed_df = load_token_feed(feed_path)

# Static Persona 
persona = TraderPersona(
    name="Alex",
    style="Real-Time Onchain Analyst",
    risk="Medium",
    experience="Intermediate",
    strategy_summary="Analyzes wallet, token, and market trends to provide trade insights.",
    preferences="Focuses on portfolio concentration, trade timing, and market moves."
)

# Build QA Agent 
if 'qa' not in st.session_state:
    st.session_state['qa'] = build_rag_agent(data_path, persona)
qa = st.session_state['qa']

# Initialize chat history
if 'chat_history' not in st.session_state:
    st.session_state['chat_history'] = []

# Trader persona summary
with st.expander("📋 Trader Persona", expanded=False):
    st.markdown(f"""
    **Name:** {persona.name}  
    **Style:** {persona.style}  
    **Risk Appetite:** {persona.risk}  
    **Experience:** {persona.experience}  
    **Strategy Summary:** {persona.strategy_summary}  
    **Preferences:** {persona.preferences}
    """)

# Display chat history 
for i, chat in enumerate(st.session_state['chat_history']):
    message(chat["question"], is_user=True, key=f"user_{i}")
    message(chat["answer"], is_user=False, key=f"bot_{i}")

# Bottom input box 
user_input = st.chat_input("Ask me about your trades, wallet, or token strategy...")

if user_input:
    with st.spinner("Thinking..."):
        try:
            result = qa.invoke({"query": user_input})
            st.write("🔍 Raw result:", result)
            if isinstance(result, dict) and "answer" in result:
                answer_text = result["answer"]
            elif isinstance(result, dict):
                answer_text = result.get("output_text", "[No output_text returned]")
            else:
                answer_text = str(result)

            st.write("📤 Final answer text:", answer_text)
            cleaned_answer = answer_text.strip()

            # Store in chat history
            st.session_state['chat_history'].append({
                "question": user_input,
                "answer": cleaned_answer
            })

        except Exception as e:
            st.error(f"Error: {str(e)}")