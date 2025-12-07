# app/main.py

from fastapi import FastAPI

from core.db import Base, engine

# Import models so SQLAlchemy knows about them before create_all()
from core.people import models as people_models    # noqa: F401
from core.memory import models as memory_models    # noqa: F401

# Route modules
from app.routes import face as face_routes
from app.routes import persons as persons_routes
from app.routes import thoughts as thoughts_routes
from app.routes import memory as memory_routes 
from app.routes import episodes as episodes_routes

# ---------------------------------------------------------------------------
# DB: create tables (dev mode). In production, use migrations instead.
# ---------------------------------------------------------------------------
Base.metadata.create_all(bind=engine)


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
app = FastAPI(title="MemorEyes API", version="0.3.0")


# ---------------------------------------------------------------------------
# Routers
# ---------------------------------------------------------------------------

# face.py already has: router = APIRouter(prefix="/face", tags=["face"])
# → so we INCLUDE WITHOUT an extra prefix here
app.include_router(face_routes.router)

# persons.py in your current setup uses router = APIRouter()
# → so we add the prefix here
app.include_router(persons_routes.router, prefix="/persons", tags=["persons"])

# thoughts.py was defined with: router = APIRouter(prefix="/thoughts", tags=["thoughts"])
# → so we include it without extra prefix
app.include_router(thoughts_routes.router)

# memory.py was defined with: router = APIRouter(prefix="/memory", tags=["memory"])
# → so we include it without extra prefix
app.include_router(memory_routes.router)

# episodes.py was defined with: router = APIRouter(prefix="/episodes", tags=["episodes"])
# → so we include it without extra prefix 
app.include_router(episodes_routes.router)

# ---------------------------------------------------------------------------
# Simple health + root endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
def health():
    return {"ok": True}


@app.get("/")
def read_root():
    return {"message": "Welcome to the MemorEyes API!"}
