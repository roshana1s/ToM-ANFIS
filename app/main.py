from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import JSONResponse

from app.core.loader import loader
from app.routes.predict import router


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI lifespan context manager.
    Loads the ANFIS model and scaler at startup.
    """
    print("\n" + "="*70)
    print("ToM-ANFIS API Startup")
    print("="*70)
    
    try:
        loader.load()
        print("="*70 + "\n")
        print("✓ Model loader initialised — API ready to serve")
        print("✓ Visit http://localhost:8000/docs for API documentation\n")
    except Exception as e:
        print(f"\n✗ STARTUP FAILED: {str(e)}\n")
        print("="*70)
        raise
    
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

app.include_router(router, prefix="/predict", tags=["Prediction"])


@app.get("/", tags=["Health"])
async def root():
    """API root endpoint."""
    return {
        "service": "ToM-ANFIS Pedestrian Crossing Intention API",
        "version": "1.0.0",
        "docs": "http://localhost:8000/docs"
    }


@app.get("/health", tags=["Health"])
async def health():
    """Health check endpoint."""
    return {"status": "healthy"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=False)