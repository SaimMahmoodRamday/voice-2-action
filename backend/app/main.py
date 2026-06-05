from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api.routes import router

app = FastAPI(title="Voice2Action", version="0.1.0")


# Outermost middleware: an unhandled exception in a route would otherwise be
# turned into a 500 by Starlette's error handler, which sits *outside* the CORS
# middleware — so the browser can't read it and reports "Failed to fetch". This
# catches it and returns a 500 with the CORS header attached.
@app.middleware("http")
async def cors_safe_errors(request: Request, call_next):
    try:
        return await call_next(request)
    except Exception as exc:  # noqa: BLE001 — deliberate catch-all safety net
        return JSONResponse(
            status_code=500,
            content={"detail": f"Internal server error: {exc}"},
            headers={"Access-Control-Allow-Origin": "*"},
        )


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)
