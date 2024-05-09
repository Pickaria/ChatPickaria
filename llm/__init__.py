def _get_ollama(model: str, embeddings_model: str):
    from langchain_community.llms import Ollama
    from langchain_community.embeddings import OllamaEmbeddings

    llm = Ollama(model=model)
    embeddings = OllamaEmbeddings(model=embeddings_model)

    return llm, embeddings


def _get_openai(model: str, embeddings_model: str):
    import os

    api_key = os.environ["OPENAI_API_KEY"]

    from langchain_openai import ChatOpenAI, OpenAIEmbeddings

    llm = ChatOpenAI(api_key=api_key)
    embeddings = OpenAIEmbeddings(model=embeddings_model)

    return llm, embeddings


def get_llm(provider: str, model: str, embeddings_model: str | None = None):
    assert provider in ["ollama", "openai"], "Provider is invalid"

    if embeddings_model is None:
        embeddings_model = model

    if provider == "ollama":
        return _get_ollama(model, embeddings_model)
    elif provider == "openai":
        return _get_openai(model, embeddings_model)
