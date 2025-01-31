import logging
import time
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, HTTPException, Request, Response
from pydantic import HttpUrl

import RAG
from schemas.request import PredictionRequest, PredictionResponse
from utils.logger import setup_logging

log = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    setup_logging()
    yield

app = FastAPI(lifespan=lifespan)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()

    body = await request.body()
    log.info(
        f"Incoming request: {request.method} {request.url}\n"
        f"Request body: {body.decode()}"
    )

    response = await call_next(request)
    process_time = time.time() - start_time

    response_body = b""
    async for chunk in response.body_iterator:
        response_body += chunk

    log.info(
        f"Request completed: {request.method} {request.url}\n"
        f"Status: {response.status_code}\n"
        f"Response body: {response_body.decode()}\n"
        f"Duration: {process_time:.3f}s"
    )

    return Response(
        content=response_body,
        status_code=response.status_code,
        headers=dict(response.headers),
        media_type=response.media_type,
    )

@app.get('/')
def test():
    return 'Working!'

@app.post("/api/request", response_model=PredictionResponse)
def predict(body: PredictionRequest):
    log.info(f"Processing prediction request with id: {body.id}")
    try:
        # Здесь будет вызов вашей модели
        result = RAG.answer(body.query)
        sources  =  [HttpUrl(d.metadata['source']) for d in  result['context']]
        answer = result['answer']
        response = PredictionResponse(
            id=body.id,
            answer=answer.answer,
            reasoning=RAG.ai_model+': '+answer.reasoning,
            sources=sources,
        )
        log.info(f"Successfully processed request {body.id}")
        return response
    except ValueError as e:
        error_msg = str(e)
        log.error(f"Validation error for request {body.id}: {error_msg}")
        raise HTTPException(status_code=400, detail=error_msg)
    except Exception as e:
        log.error(f"Internal error processing request {body.id}: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)

