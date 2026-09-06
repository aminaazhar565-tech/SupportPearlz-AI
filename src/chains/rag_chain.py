import time
from langchain_openai import ChatOpenAI
from src.config import settings
from src.chains.prompts import RAG_PROMPT
from src.chains.schemas import SupportResponse, Confidence
from src.chains.memory import condense_question, format_history
from src.retrieval.retriever import retrieve
from src.retrieval.vector_store import index_exists
from src.utils.logging_setup import setup_logging

logger = setup_logging(settings.log_dir)

def _context(docs):
    blocks = []
    for i, d in enumerate(docs, 1):
        source = d.metadata.get("source", "unknown")
        location = d.metadata.get("location", "unknown")
        score = d.metadata.get("retrieval_score", 0)
        blocks.append(f"[S{i}] source={source} | location={location} | score={score}\n{d.page_content}")
    return "\n\n---\n\n".join(blocks)

def answer_question(question: str, history: list[dict], api_key: str):
    start = time.perf_counter()
    if not api_key:
        return SupportResponse(
            answer="Please enter your OpenAI API key in the sidebar to start chatting.",
            sources=[], confidence=Confidence.NONE, answered=False
        ), "", []

    if not index_exists():
        return SupportResponse(
            answer="The knowledge base is not built yet. Please click “Build / Refresh Knowledge Base” in the sidebar.",
            sources=[], confidence=Confidence.NONE, answered=False
        ), question, []

    try:
        rewritten = condense_question(question, history, api_key, settings.openai_model)
        docs = retrieve(rewritten, settings.retrieval_k, settings.score_threshold)
        logger.info("query=%r | rewritten=%r | retrieved=%s", question, rewritten,
                    [(d.metadata.get("source"), d.metadata.get("retrieval_score")) for d in docs])

        if not docs:
            result = SupportResponse(
                answer="I couldn't find supporting information for that in the Pearlz documentation. Please contact human support for confirmation.",
                sources=[], confidence=Confidence.NONE, answered=False
            )
            return result, rewritten, []

        llm = ChatOpenAI(api_key=api_key, model=settings.openai_model, temperature=settings.temperature)
        chain = RAG_PROMPT | llm.with_structured_output(SupportResponse)
        result = chain.invoke({
            "history": format_history(history),
            "context": _context(docs),
            "question": question,
        })

        valid_sources = set()
        for i, d in enumerate(docs, 1):
            valid_sources.add(f"[S{i}] {d.metadata.get('source')} — {d.metadata.get('location')}")
        result.sources = [
            s for s in result.sources
            if any(s in candidate or candidate.startswith(s) for candidate in valid_sources)
        ]
        # Ensure citations are never empty for a grounded answer.
        if result.answered and not result.sources:
            result.sources = list(valid_sources)[:1]

        latency = round(time.perf_counter() - start, 2)
        logger.info("latency=%ss | refusal=%s", latency, not result.answered)
        return result, rewritten, docs

    except Exception as exc:
        logger.exception("query failed")
        result = SupportResponse(
            answer=f"I couldn't complete that request safely. Please try again. Technical detail: {type(exc).__name__}.",
            sources=[], confidence=Confidence.NONE, answered=False
        )
        return result, question, []
