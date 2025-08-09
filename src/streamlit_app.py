import streamlit as st
import requests

st.set_page_config(page_title="RAG Chat with Ollama", layout="wide")
st.title("📚 Retrieval-Augmented Generation (RAG) with Ollama")

# --- Session State Setup ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- Chat display ---
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# --- User input ---
if prompt := st.chat_input("Ask something..."):
    # Show user message
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Prepare payload
    payload = {"question": prompt}

    try:
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                res = requests.post(
                    "http://localhost:5000/ask", json=payload, timeout=60
                )
                res.raise_for_status()
                result = res.json()

                answer_data = result.get("answer", {})
                content = answer_data.get("content", "").strip()

                # --- Optional: Split <think> block ---
                if content.startswith("<think>"):
                    think_end = content.find("</think>")
                    thinking = content[7:think_end].strip()
                    final_answer = content[think_end + 8:].strip()

                    with st.expander("🤔 Internal Reasoning", expanded=False):
                        st.markdown(thinking)

                    st.markdown(final_answer)
                    st.session_state.messages.append(
                        {"role": "assistant", "content": final_answer}
                    )
                else:
                    st.markdown(content)
                    st.session_state.messages.append(
                        {"role": "assistant", "content": content}
                    )

                # --- Show source documents if present ---
                sources = answer_data.get("source_documents", [])
                if sources:
                    with st.expander("📄 Sources", expanded=False):
                        for i, src in enumerate(sources, 1):
                            st.markdown(f"**Source {i}:**")
                            st.markdown(src.get("text", ""))
                            meta = src.get("metadata", {})
                            if meta:
                                st.caption(f"Metadata: {meta}")

    except Exception as e:
        st.error(f"Error: {e}")
