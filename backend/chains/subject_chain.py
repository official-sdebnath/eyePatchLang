from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI


def make_physics_chain():
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You are a physics teacher assistant called EyePatch."),
            ("user", "{question}"),
        ]
    )
    model = ChatOpenAI(model_name="gpt-4o-mini", temperature=0.2)
    return prompt | model


def make_math_chain():
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You are a math assistant called EyePatch."),
            ("user", "{question}"),
        ]
    )
    model = ChatOpenAI(model_name="gpt-4o-mini", temperature=0.2)
    return prompt | model
