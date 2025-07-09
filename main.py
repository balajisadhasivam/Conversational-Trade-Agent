import streamlit as st
from agent.persona import assign_persona_from_trades
from agent.chat_agent import build_rag_agent
from agent.data_utils import load_trades, generate_synthetic_trades
import os
from dotenv import load_dotenv
import logging
import re

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

# Add a Clear Chat button
if st.button("Clear Chat"):
    st.session_state['chat_history'] = []

# Enhanced cleaning to avoid sentence-level repetition
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

with st.form(key="qa_form", clear_on_submit=True):
    user_input = st.text_input("Your question", key="user_input")
    submit = st.form_submit_button("Ask")
    if submit and user_input.strip():
        logging.info(f"User question: {user_input}")
        with st.spinner("Thinking..."):
            result = qa.invoke(user_input)
            logging.info(f"Agent response: {result['answer']}")
            cleaned_answer = clean_response_to_single_paragraph(result['answer'])
            st.session_state['chat_history'].append({
                "question": user_input,
                "answer": cleaned_answer
            })

# Display chat history
for chat in st.session_state['chat_history']:
    st.markdown(f"**You:** {chat['question']}")
    st.markdown(f"**Trader:** {chat['answer']}")
