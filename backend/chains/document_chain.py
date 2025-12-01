from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains.combine_documents import create_stuff_documents_chain


def document_chain():
    """
    Document combining chain.
    Takes a list of Documents and produces a single 'stuffed' context string.
    """
    # This prompt MUST include a {context} placeholder.
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You are EyePatch, a helpful document-processing assistant."),
            (
                "user",
                "Use the following context to help answer questions:\n\n{context}\n\n"
                "User question:\n{question}",
            ),
        ]
    )

    model = ChatOpenAI(model_name="gpt-4o-mini", temperature=0.2)
    return create_stuff_documents_chain(llm=model, prompt=prompt)
