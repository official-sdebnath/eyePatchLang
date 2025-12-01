import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

model = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.7)

# -------------- LangChain Setup --------------

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a teacher assistant called EyePatch. Your job is to answer any question regarding uploaded documents.",
        ),
        ("human", "{question}"),
    ]
)

output_parsers = StrOutputParser()

chain = prompt | model | output_parsers

# -------------- Streamlit App --------------

st.title("EyePatch - Using Langchain")
input_text = st.text_input("Ask a question about your document:")

if input_text:
    st.write(chain.invoke(input_text))
