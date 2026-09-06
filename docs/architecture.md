# SupportPearlz Architecture

## Offline ingestion path
Knowledge Base -> LangChain loaders -> normalization -> RecursiveCharacterTextSplitter -> local Sentence-Transformers embeddings -> persistent Chroma index.

## Online query path
User question + bounded history -> query condensation -> Chroma semantic retrieval -> relevance gate -> labelled context -> grounded LangChain prompt -> OpenAI structured output -> citation validation -> UI.

The two paths are intentionally separate so normal queries do not re-embed the knowledge base.
