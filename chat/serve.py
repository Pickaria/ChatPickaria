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
