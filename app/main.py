
from fastapi import FastAPI

from app.api.routes import router
from app.core.config import get_settings


settings = get_settings()

app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description=(
        "A document-grounded question-answering API "
        "with semantic retrieval and source evidence."
    ),
)

app.include_router(router)


@app.get("/health", tags=["Health"])
def health_check() -> dict[str, str]:
    return {"status": "ok"}
