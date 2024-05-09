def create_chain():
    import os

    provider = os.environ.get("PROVIDER")
    model = os.environ.get("MODEL")
    embeddings_model = os.environ.get("EMBEDDINGS_MODEL")

    from chat.llm import get_llm

    llm, embeddings = get_llm(provider, model, embeddings_model)

    from langchain_core.prompts import ChatPromptTemplate

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """Tu es un assistant virtuel sur le serveur survie Minecraft nommé Pickaria.
Ton rôle est d'aider les autres joueurs en donnant une réponse claire en quelques mots.
Ta réponse doit utiliser le tutoiement.
Voici un extrait de la documentation du serveur pour pouvoir répondre à la question :
{context}""",
            ),
            ("user", "Question : {input}"),
        ]
    )

    # Load documents
    from langchain_community.document_loaders import UnstructuredFileLoader
    from chat.documents import get_document_paths

    loader = UnstructuredFileLoader(
        get_document_paths(), strategy="fast", mode="single"
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
