from langchain_openai import ChatOpenAI
from src.chains.prompts import CONDENSE_PROMPT

def format_history(history: list[dict], max_turns: int = 6) -> str:
    recent = history[-max_turns:]
    return "\n".join(f"{m['role']}: {m['content']}" for m in recent)

def condense_question(question: str, history: list[dict], api_key: str, model_name: str) -> str:
    if not history:
        return question
    llm = ChatOpenAI(api_key=api_key, model=model_name, temperature=0)
    prompt = CONDENSE_PROMPT.format_messages(
        history=format_history(history),
        question=question
    )
    response = llm.invoke(prompt)
    return response.content.strip()
