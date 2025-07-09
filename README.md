## Setup Instructions

1. **Create and activate a virtual environment:**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the agent:**
   ```bash
   streamlit run main.py
   ```

---

## Project Overview

This project implements a Retrieval-Augmented Generation (RAG) pipeline using the **Llama-3.3-70B-Instruct-Turbo** large language model, with **synthetic data** as the retrieval corpus.

### 1. Large Language Model (LLM) Used

- **Model:** `meta-llama/Llama-3.3-70B-Instruct-Turbo`
- **Provider:** Meta (Together.ai API)
- **Purpose:**  
  The LLM generates natural language responses to user queries, leveraging context retrieved from a synthetic knowledge base.

### 2. Embedding Model

- **Model:** `sentence-transformers/all-MiniLM-L6-v2`
- **Purpose:**  
  This model is used to convert both user queries and knowledge base entries into vector embeddings, enabling efficient semantic search and retrieval.

### 3. Retrieval-Augmented Generation (RAG) Pipeline

- **Retrieval:**  
  When a user submits a query, the system uses the embedding model to find the most relevant synthetic data entries from the knowledge base.
- **Augmentation:**  
  The retrieved context is provided to the LLM as additional input.
- **Generation:**  
  The LLM combines its own knowledge with the retrieved context to generate a final, contextually grounded answer.

**Benefits:**  
- Reduces hallucination by grounding answers in real (synthetic) data  
- Enables domain-specific and up-to-date responses

### 4. Synthetic Data

- **Purpose:**  
  Synthetic data is artificially generated to simulate real-world documents or FAQs, allowing for safe and controlled testing of the RAG pipeline.
- **Usage:**  
  The synthetic data is stored in the `data/` directory and serves as the retrieval corpus for the RAG pipeline.

---

**Example Workflow:**

1. User asks a question via the Streamlit UI.
2. The system retrieves relevant synthetic data entries using semantic search.
3. The LLM receives the query and retrieved context, then generates a response.
4. The answer is displayed to the user.

---