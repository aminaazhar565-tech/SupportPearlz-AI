import json
from pathlib import Path
from src.chains.rag_chain import answer_question

ROOT = Path(__file__).resolve().parents[1]
cases = json.loads((ROOT/"evaluation/test_questions.json").read_text(encoding="utf-8"))

def run(api_key: str):
    rows = []
    history = []
    for case in cases:
        result, rewritten, docs = answer_question(case["question"], history, api_key)
        rows.append({
            "id": case["id"],
            "category": case["category"],
            "question": case["question"],
            "answer": result.answer,
            "sources": result.sources,
            "confidence": result.confidence.value,
            "answered": result.answered,
            "expected": case["expected"],
            "rewritten_query": rewritten,
        })
        history.extend([
            {"role":"user","content":case["question"]},
            {"role":"assistant","content":result.answer},
        ])
    out = ROOT/"evaluation/results"
    out.mkdir(parents=True, exist_ok=True)
    (out/"results.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"Wrote {len(rows)} results to {out/'results.json'}")

if __name__ == "__main__":
    key = input("OpenAI API key: ").strip()
    run(key)
