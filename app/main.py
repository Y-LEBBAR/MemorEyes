from fastapi import FastAPI
from app.routes.face import router as face_router
app = FastAPI(title="MemorEyes API")

app.include_router(face_router)

@app.get("/health")
def health():
    return {"ok": True}

@app.get("/")
def read_root():
    return {"message": "Welcome to the MemorEyes API!"}