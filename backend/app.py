from fastapi import FastAPI
from api.v1.router import router as v1_router


app = FastAPI(
	title="AnimGen API",
	version="0.1.0",
)

app.include_router(v1_router, prefix="/api/v1")


@app.get("/")
async def root() -> dict[str, str]:
	return {"name": "AnimGen API", "status": "running"}


@app.get("/health")
async def health_check() -> dict[str, str]:
	return {"status": "ok"}
