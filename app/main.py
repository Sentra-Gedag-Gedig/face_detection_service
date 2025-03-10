from fastapi import FastAPI
from app.routes.detection import router as detection_router

app = FastAPI(title="Face Detection Service")
app.include_router(detection_router, prefix="/api/v1/detect",
                   tags=["detection"])
