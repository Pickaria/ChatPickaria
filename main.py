def create_chain():
    import os

    provider = os.environ.get("PROVIDER")
    model = os.environ.get("MODEL")
    embeddings_model = os.environ.get("EMBEDDINGS_MODEL")

    from llm import get_llm

    llm, embeddings = get_llm(provider, model, embeddings_model)

    from langchain_core.prompts import ChatPromptTemplate

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "Tu es un joueur expérimenté de Minecraft jouant sur le serveur survie Pickaria. "
                "Ton rôle est d'aider les autres joueurs en donnant une réponse claire en quelques mots. {context}",
            ),
            ("user", "{input}"),
        ]
    )

    # Load documents
    from langchain_community.document_loaders import UnstructuredFileLoader

    loader = UnstructuredFileLoader(
        ["./docs/jobs.md", "./docs/rewards.md"], strategy="fast", mode="single"
    )

    docs = loader.load()

    # Build indexes
    from langchain_community.vectorstores import FAISS
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    text_splitter = RecursiveCharacterTextSplitter()
    documents = text_splitter.split_documents(docs)
    vector = FAISS.from_documents(documents, embeddings)

    # Document chain
    from langchain.chains.combine_documents import create_stuff_documents_chain
    from langchain_core.output_parsers import StrOutputParser

    output_parser = StrOutputParser()
    document_chain = create_stuff_documents_chain(
        llm, prompt, output_parser=output_parser
    )

    # Document retrieval
    from langchain.chains import create_retrieval_chain

    retriever = vector.as_retriever()
    retrieval_chain = create_retrieval_chain(retriever, document_chain)

    return retrieval_chain


def serve(chain):
    from fastapi import FastAPI, Request, Response, status
    from langchain.pydantic_v1 import BaseModel
    from langserve import add_routes
    import os

    app = FastAPI(
        title="LangChain Server",
        version="1.0",
        description="A simple API server using LangChain's Runnable interfaces",
    )

    # Authentication middleware
    x_api_key = os.environ.get("X_API_KEY")

    @app.middleware("http")
    async def add_process_time_header(request: Request, call_next):
        if x_api_key is not None:
            if "x-api-key" not in request.headers:
                return Response(status_code=status.HTTP_400_BAD_REQUEST)
            if request.headers["x-api-key"] != x_api_key:
                return Response(status_code=status.HTTP_401_UNAUTHORIZED)
        response = await call_next(request)
        return response

    class Input(BaseModel):
        input: str

    class Output(BaseModel):
        input: str
        answer: str

    add_routes(
        app,
        chain.with_types(input_type=Input, output_type=Output),
    )

    return app


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(serve(create_chain()), host="0.0.0.0", port=8000)
