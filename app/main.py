from fastapi import FastAPI
app = FastAPI(title="MemorEyes API")

@app.get("/health")
def health():
    return {"ok": True}
