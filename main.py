import uvicorn

from chat import serve, create_chain

if __name__ == "__main__":
    chain = create_chain()
    uvicorn.run(serve(chain), host="0.0.0.0", port=8000)
