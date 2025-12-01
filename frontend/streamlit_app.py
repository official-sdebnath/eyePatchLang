import requests
import streamlit as st

BACKEND_URL = "http://localhost:8000"
INGEST_URL = f"{BACKEND_URL}/ingest"
RAG_QUERY_URL = f"{BACKEND_URL}/rag/query"

st.set_page_config(page_title="EyePatch", page_icon="🧐")
st.title("EyePatch — upload, select subject, ask")

# 1) file upload + subject + question inputs (single flow)
uploaded = st.file_uploader("Upload a PDF to ask about", type=["pdf"])
subject = st.selectbox("Subject", ["physics", "math"])
question = st.text_input("Question about the uploaded document")

# 2) Button that runs ingest then query in one go
if st.button("Ingest & Ask"):
    if not uploaded:
        st.error("Please upload a PDF first.")
    elif not question.strip():
        st.error("Please enter a question.")
    else:
        with st.spinner("Uploading PDF and creating index..."):
            try:
                files = {
                    "file": (uploaded.name, uploaded.getvalue(), "application/pdf")
                }
                resp = requests.post(INGEST_URL, files=files, timeout=180)
                resp.raise_for_status()
                ingest_data = resp.json()
                # backend returns index_path internally; we don't show it
                index_path = ingest_data.get("index_path")
                if not index_path:
                    st.error("Ingest succeeded but no index information returned.")
                else:
                    st.success("Ingest complete — querying the index now...")
                    # call RAG
                    payload = {
                        "index_path": index_path,
                        "subject": subject,
                        "question": question,
                        "k": 4,
                    }
                    qresp = requests.post(RAG_QUERY_URL, json=payload, timeout=60)
                    qresp.raise_for_status()
                    out = qresp.json()
                    answer = out.get("answer") or out.get("result") or out.get("text")
                    sources = out.get("sources", [])
                    st.subheader("Answer")
                    st.write(answer)
                    if sources:
                        st.subheader("Sources")
                        for s in sources:
                            src = s.get("source", "unknown")
                            chunk = s.get("chunk")
                            st.write(f"- {src} (chunk {chunk})")
            except requests.RequestException as e:
                st.error(f"Request failed: {e}")
                try:
                    st.text(resp.text)
                except Exception:
                    pass

st.markdown("---")
st.caption(
    "Notes: this runs ingestion synchronously; for large files consider background ingestion on the server."
)
