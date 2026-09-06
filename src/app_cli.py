from src.config import settings
from src.chains.rag_chain import answer_question

def main():
    key = input("OpenAI API key: ").strip()
    history = []
    print("SupportPearlz CLI. Type /reset to clear history or /exit to quit.")
    while True:
        q = input("\nYou > ").strip()
        if q == "/exit":
            break
        if q == "/reset":
            history = []
            print("Session reset.")
            continue
        result, _, _ = answer_question(q, history, key)
        print(f"\nBot > {result.answer}")
        if result.sources:
            print("Sources:")
            for source in result.sources:
                print(f"  {source}")
        print(f"Confidence: {result.confidence.value}")
        history.extend([{"role":"user","content":q},{"role":"assistant","content":result.answer}])

if __name__ == "__main__":
    main()
