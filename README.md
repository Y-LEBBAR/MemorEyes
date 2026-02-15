# MemorEyes

MemorEyes is an AI-powered memory assistant designed for augmented reality. This MVP provides the foundational backend for face recognition, speaker detection, profile creation, and memory retrieval. The backend is built using FastAPI and is designed to support an AR headset client.

## Features (MVP Level)

* Health check endpoint to verify the API is running
* Clean project structure for modular expansion
* Virtual environment and dependency management

## Future Capabilities

* Face detection and embeddings
* Speaker activity detection
* Identity binding from conversational signals
* Long-term memory storage (SQLite / FAISS)
* AR overlay support for live identification

## Project Structure

```
MemorEyes/
│
├── app/                  # FastAPI routes
│   └── main.py           # Main API entry point
│
├── core/                 # Core logic modules (future)
├── data/                 # Local data (ignored by Git)
├── docs/                 # Documentation (architecture + API)
├── tests/                # Unit tests
│
├── requirements.txt      # Project dependencies
├── README.md             # Project overview and usage
└── .gitignore            # Git ignore rules
```

Additional architecture and roadmap details live in [`docs/system_overview.md`](docs/system_overview.md).

## Setup Instructions

### 1. Create and activate a virtual environment

```bash
python -m venv .venv
.\.venv\Scripts\activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

## Running the API

Use Uvicorn to start the development server:

```bash
uvicorn app.main:app --reload
```

Once running, visit:

```
http://127.0.0.1:8000/health
```

Response:

```json
{"ok": true}
```

## API Documentation

FastAPI automatically generates interactive API documentation.

Swagger UI:

```
http://127.0.0.1:8000/docs
```

ReDoc:

```
http://127.0.0.1:8000/redoc
```

## Development Notes

* Always activate the virtual environment before working.
* Use `pip install <package>` then update the project requirements with:

```bash
pip freeze > requirements.txt
```

* Push frequently to GitHub using:

```bash
git add .
git commit -m "your message"
git push origin main
```

## Roadmap

* Implement face detection pipeline
* Implement facial embeddings + identity database
* Add audio-based speaker detection
* Create memory store and retrieval API
* Connect to AR headset interface

## License

This project is currently private and unlicensed until further decision.
