"""Step 3: a simple chat web app. Run with: streamlit run app.py"""
from pathlib import Path

import streamlit as st

st.set_page_config(page_title="Irish R&D Reports Chatbot")
st.title("Irish R&D Reports Chatbot")
st.caption(
    "Ask questions about public reports on innovation and R&D in Ireland. "
    "Every answer cites the report and page it came from."
)

if not Path("index/chunks.json").exists():
    st.error("No index found. Run `python ingest.py` first, then refresh this page.")
    st.stop()

from rag import answer  # imported after the check, because it loads the index

if "history" not in st.session_state:
    st.session_state.history = []

for role, text in st.session_state.history:
    with st.chat_message(role):
        st.markdown(text)

question = st.chat_input("e.g. How much did businesses spend on R&D in Ireland?")
if question:
    st.session_state.history.append(("user", question))
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Searching the reports..."):
            reply, sources, tool_used = answer(question)
        st.markdown(reply)
        if tool_used:
            st.caption("Used the R&D statistics lookup tool for an exact figure.")
        with st.expander("Sources"):
            for s in sources:
                st.markdown(
                    f"**{s['source']}, page {s['page']}** (similarity {s['score']:.2f})\n\n"
                    f"> {s['text'][:300]}..."
                )
    st.session_state.history.append(("assistant", reply))