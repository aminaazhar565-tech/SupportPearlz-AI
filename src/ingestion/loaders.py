from pathlib import Path
import hashlib
import re
from typing import List
import pandas as pd
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader, TextLoader, UnstructuredWordDocumentLoader

def _clean(text: str) -> str:
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()

def load_file(path: Path) -> List[Document]:
    ext = path.suffix.lower()
    if ext == ".pdf":
        docs = PyPDFLoader(str(path)).load()
    elif ext in {".docx", ".doc"}:
        docs = UnstructuredWordDocumentLoader(str(path)).load()
    elif ext in {".md", ".txt"}:
        docs = TextLoader(str(path), encoding="utf-8").load()
    elif ext == ".csv":
        df = pd.read_csv(path)
        docs = [Document(page_content=df.to_csv(index=False), metadata={})]
    else:
        raise ValueError(f"Unsupported file type: {ext}")

    out = []
    for i, d in enumerate(docs):
        content = _clean(d.page_content)
        if not content:
            continue
        meta = dict(d.metadata)
        meta.update({
            "source": path.name,
            "doc_type": _doc_type(path.name),
            "version": "1.0",
            "last_updated": "2026-09-01",
        })
        if ext == ".pdf":
            meta["location"] = f"page {int(meta.get('page', i)) + 1}"
        else:
            meta["location"] = "document"
        d.page_content = content
        d.metadata = meta
        out.append(d)
    return out

def _doc_type(name: str) -> str:
    n = name.lower()
    if "pricing" in n: return "pricing"
    if "warranty" in n or "refund" in n or "shipping" in n or "privacy" in n: return "policy"
    if "troubleshooting" in n: return "guide"
    if "installation" in n: return "guide"
    if "service" in n: return "agreement"
    if "faq" in n: return "faq"
    if "manual" in n: return "manual"
    return "document"

def load_knowledge_base(folder: Path):
    documents = []
    skipped = []
    seen = set()
    paths = sorted(p for p in folder.rglob("*") if p.is_file())
    for path in paths:
        try:
            docs = load_file(path)
            for d in docs:
                digest = hashlib.sha256(d.page_content.encode("utf-8")).hexdigest()
                if digest in seen:
                    continue
                seen.add(digest)
                documents.append(d)
        except Exception as exc:
            skipped.append({"file": path.name, "reason": str(exc)})
    return documents, skipped
