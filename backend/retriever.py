import os
from typing import Dict, Any
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

from chains import document_chain  # your create_stuff_documents_chain wrapper
from chains import subject_chain  # your make_physics_chain / make_math_chain

_EMBEDDINGS = OpenAIEmbeddings()


def load_retriever(index_path: str, k: int = 4):
    if not os.path.exists(index_path):
        raise FileNotFoundError(f"Index not found at {index_path}")
    # dev-only: allow_pickle for local index you created
    db = FAISS.load_local(index_path, _EMBEDDINGS, allow_dangerous_deserialization=True)
    return db.as_retriever(search_kwargs={"k": k})


def _call_retriever(retriever, query: str, k: int):
    # Preferred: if retriever is an LCEL Runnable, call invoke with common keys
    if hasattr(retriever, "invoke") and callable(retriever.invoke):
        for key in ("query", "input", "question"):
            try:
                out = retriever.invoke({key: query})
                # normalize common shapes
                if isinstance(out, list):
                    return out
                if isinstance(out, dict):
                    for docs_key in ("documents", "results", "docs"):
                        if docs_key in out and isinstance(out[docs_key], list):
                            return out[docs_key]
                if hasattr(out, "page_content"):
                    return [out]
            except Exception:
                # ignore and try next shape
                pass

    # Fallback public methods (some versions)
    if hasattr(retriever, "get_relevant_documents"):
        return retriever.get_relevant_documents(query)
    if hasattr(retriever, "retrieve"):
        return retriever.retrieve(query)

    # Last resort: use underlying vectorstore similarity_search
    if hasattr(retriever, "vectorstore") and hasattr(
        retriever.vectorstore, "similarity_search"
    ):
        return retriever.vectorstore.similarity_search(query, k=k)

    raise RuntimeError(
        "No supported retrieval method found on retriever for this LangChain version."
    )


def choose_subject_chain(subject: str):
    s = subject.lower().strip()
    if s == "physics":
        return subject_chain.make_physics_chain()
    if s == "math":
        return subject_chain.make_math_chain()
    raise ValueError(f"Unknown subject: {subject}")


def run_rag(index_path: str, subject: str, question: str, k: int = 4) -> Dict[str, Any]:
    retriever = load_retriever(index_path, k=k)
    docs = _call_retriever(retriever, question, k)

    # docs should be List[Document] now
    # 1) combine docs using your document_chain (create_stuff_documents_chain)
    doc_chain_callable = (
        document_chain.document_chain()
    )  # your function returning the chain
    # doc_chain expects {"context": docs, "question": "..."}
    combined_out = doc_chain_callable.invoke({"context": docs, "question": question})

    # normalize combined_out -> text (the doc_chain may return str or dict or message-like)
    if hasattr(combined_out, "content"):
        context_text = combined_out.content
    elif isinstance(combined_out, dict):
        # try common keys
        context_text = (
            combined_out.get("text") or combined_out.get("result") or str(combined_out)
        )
    else:
        context_text = str(combined_out)

    # 2) call your subject chain, feeding the combined context and the question
    subj = choose_subject_chain(subject)

    # if your subject chain supports {context, question}, call:
    try:
        out = subj.invoke({"context": context_text, "question": question})
    except Exception:
        # fallback: subject chain only takes {question}
        combined_question = f"Context:\n{context_text}\n\nQuestion:\n{question}"
        out = subj.invoke({"question": combined_question})

    # normalize final output
    if hasattr(out, "content"):
        answer = out.content
    elif isinstance(out, dict):
        answer = out.get("answer") or out.get("result") or out.get("text") or str(out)
    else:
        answer = str(out)

    sources = [
        {"source": d.metadata.get("source"), "chunk": d.metadata.get("chunk")}
        for d in docs
    ]
    return {"answer": answer, "sources": sources}
