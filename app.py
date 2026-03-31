import asyncio
import json
import subprocess
from typing import Optional

import ollama
from fastapi import FastAPI, Query
from fastapi.responses import StreamingResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from src.embeddings import EmbeddingGenerator
from src.db_handler import ChromaDBHandler
from src.generator import OllamaGenerator
from src.reference_resolver import resolve_references
from src.config import (
    N_RESULTS, SIMILARITY_THRESHOLD, MIN_CHUNK_SIZE, ENABLE_RERANKING,
    LLM_MODEL_NAME, LLM_MAX_TOKENS, LLM_SYSTEM_PROMPT, RESOLVE_REFERENCES,
)

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")

embedding_generator = EmbeddingGenerator()
db_handler = ChromaDBHandler()
generator = OllamaGenerator()
_async_client = ollama.AsyncClient()

session: dict = {
    "initial_topic": None,
    "chat_history": [],
    "cached_contexts": [],
}


class ChatRequest(BaseModel):
    query: str
    reset: bool = False
    collection: Optional[str] = None


@app.get("/")
async def index():
    with open("static/index.html", "r") as f:
        return HTMLResponse(f.read())


@app.post("/api/chat")
async def chat(request: ChatRequest):
    async def event_stream():
        # Handle collection switching if requested
        if request.collection:
            db_handler.switch_collection(request.collection)

        is_followup = (
            not request.reset
            and session["initial_topic"] is not None
        )

        if not is_followup:
            session["initial_topic"] = request.query
            session["chat_history"].clear()
            session["cached_contexts"].clear()

            loop = asyncio.get_event_loop()
            query_embedding = await loop.run_in_executor(
                None, embedding_generator.generate_query_embedding, request.query
            )
            results = await loop.run_in_executor(
                None,
                lambda: db_handler.query(
                    query_embedding,
                    n_results=N_RESULTS,
                    similarity_threshold=SIMILARITY_THRESHOLD,
                    min_chunk_size=MIN_CHUNK_SIZE,
                    rerank_by_section=bool(ENABLE_RERANKING),
                    prefer_headers=bool(ENABLE_RERANKING),
                ),
            )

            contexts = []
            if results and results.get("documents") and results["documents"][0]:
                for i, (doc, metadata, distance) in enumerate(
                    zip(
                        results["documents"][0],
                        results["metadatas"][0],
                        results["distances"][0],
                    )
                ):
                    similarity = db_handler._convert_distance_to_similarity(distance)
                    contexts.append(
                        {
                            "content": doc,
                            "file_path": metadata["file_path"],
                            "start_line": metadata["start_line"],
                            "end_line": metadata["end_line"],
                            "page_number": metadata.get("page_number", 1),
                            "similarity": round(similarity, 4),
                            "source_num": i + 1,
                        }
                    )

            session["cached_contexts"] = contexts

            # Resolve cross-references in retrieved chunks
            if RESOLVE_REFERENCES and contexts:
                contexts = await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: resolve_references(
                        contexts, embedding_generator, db_handler
                    ),
                )
                session["cached_contexts"] = contexts
        else:
            contexts = [
                {**ctx, "source_num": i + 1}
                for i, ctx in enumerate(session["cached_contexts"])
            ]

        yield f"event: contexts\ndata: {json.dumps(contexts)}\n\n"

        prompt = generator.construct_prompt(
            request.query,
            contexts,
            session["chat_history"],
            session["initial_topic"],
        )

        full_response = ""
        response_gen = await _async_client.generate(
            model=LLM_MODEL_NAME,
            prompt=prompt,
            system=LLM_SYSTEM_PROMPT,
            stream=True,
            options={"num_predict": LLM_MAX_TOKENS, "reasoning_effort": "high"},
        )
        async for chunk in response_gen:
            token = chunk.get("response", "")
            if token:
                full_response += token
                yield f"event: token\ndata: {json.dumps(token)}\n\n"

        session["chat_history"].append({"role": "user", "content": request.query})
        session["chat_history"].append({"role": "assistant", "content": full_response})

        yield "event: done\ndata: {}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@app.post("/api/reset")
async def reset_session():
    session["initial_topic"] = None
    session["chat_history"].clear()
    session["cached_contexts"].clear()
    return {"status": "ok"}


@app.get("/api/open-pdf")
async def open_pdf(path: str = Query(...)):
    subprocess.Popen(["open", path])


@app.get("/api/file")
async def get_file_content(path: str = Query(...)):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return {"content": f.read()}
    except Exception as e:
        return {"error": str(e)}, 404
    return {"status": "ok"}
