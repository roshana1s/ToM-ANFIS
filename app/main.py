from contextlib import asynccontextmanager

from fastapi import FastAPI

# from app.core.loader import loader
# from app.routes.predict import router


@asynccontextmanager
async def lifespan(app: FastAPI):
    # loader.load()
    yield


app = FastAPI(
    title="ToM-ANFIS Pedestrian Crossing Intention API",
    description=(
        "Predicts pedestrian crossing intention using a Theory of Mind-based "
        "Adaptive Neuro-Fuzzy Inference System trained on the PIE dataset. "
        "Returns a crossing probability and explainable IF-THEN fuzzy rules."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# app.include_router(router, prefix="/predict", tags=["Prediction"])


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=False)