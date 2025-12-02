# MemorEyes System Overview

## 1. Vision and Constraints
- **Purpose:** Augmented reality memory augmentation with real-time face recognition, context recall, and proactive surface of high-salience cues.
- **Display constraints:** Minimal overlay (single-line cues, safety-first, no overload).
- **Data fusion:** Facial embeddings, conversational transcripts, emotional/physiological signals, location, and relationship history.
- **Deployment:** Cloud/local flexibility; strict privacy with encryption and permission layers; versioned APIs for clients (e.g., `/v1`).
- **Future integrations:** Messaging (WhatsApp/Instagram), workout/health tracking, nutrition scanning, scheduling/location triggers.

## 2. Current State (MVP codebase)
- **FastAPI shell** with health/root endpoints and router composition. Endpoint expansion should live under versioned prefixes (e.g., `/v1`).
- **Face pipeline**
  - `/face/detect` → SCRFD ONNX detection via `core.face.detector` (singleton loader, BGR conversion, configurable input size).
  - `/face/embedding` → ArcFace R100 embeddings via `core.face.embeddings` (512-D output, L2 normalization).
  - Shared image utilities for byte loading, RGB conversion, cropping, resizing in `core.utils.image`.
- **Manual scripts** for detector/embedding validation under `tests/` (non-automated).

## 3. Missing Pieces / Risks
- **API hygiene:** No versioned routes, authentication, rate limiting, or error schema standardization.
- **Configuration:** Model paths and execution providers are hardcoded; need env-driven settings and per-service virtual environments.
- **Model assets:** `models/` artifacts are not checked in; provide download/registration instructions or packaging.
- **Observability:** Logging/metrics/tracing absent; no latency/throughput/error dashboards.
- **Testing:** No automated unit/integration tests; manual scripts only. CI pipeline missing.
- **Memory system:** No data model, storage, clustering, or ranking engine implemented; no privacy/permission enforcement.
- **Edge constraints:** No offline/local-cache strategy, encryption at rest/in transit, or data minimization policy.

## 4. Execution Roadmap
1. **Platform hardening**
   - Introduce configuration layer (pydantic settings), environment-based model paths, versioned API prefix, and auth stubs (API key/OAuth placeholder).
   - Add structured logging and basic telemetry hooks.
   - Package ONNX asset bootstrap (download scripts/checksums).
2. **Face pipeline completeness**
   - Add batch endpoints, bounding-box alignment helpers, and deterministic preprocessing.
   - Implement automated tests with fixtures/mocks for detector and embedding modules.
3. **Memory & context foundation**
   - Define domain models (Memory, Person, InteractionContext) with type-safe DTOs and repositories (abstract interfaces for vector DB/local store).
   - Implement ingestion APIs for transcripts, events, and embeddings; enforce validation and privacy flags.
4. **Ranking & retrieval**
   - Build retrieval pipeline (context fusion, similarity search, re-ranking). Target <50 ms end-to-end for cached embeddings.
   - Add salience scoring, decay functions, and relationship weighting.
   - Create summarization service for long-term history snapshots.
5. **UX & integrations**
   - AR overlay formatter producing single-line cues; safety guardrails.
   - Messaging/workout/nutrition connectors behind feature flags.
6. **Reliability & compliance**
   - CI/CD, load testing, PII handling policies, encryption, permissions, audit logging.

## 5. Memory Data Model (draft)
- **Memory (aggregate root)**
  - `id: UUID`
  - `person_id: UUID | None` (link to known identity; optional for unknown faces)
  - `timestamp: datetime` (UTC)
  - `location: Optional[str]` (geo/human-readable)
  - `transcript_snippet: str` (<= 512 chars)
  - `topics: list[str]`
  - `emotion_tags: list[str]` (e.g., calm, excited, stressed)
  - `relationship_history: dict[str, float]` (lightweight scores such as trust/rapport)
  - `embedding: ndarray` (semantic embedding of snippet/context; stored in vector DB)
  - `face_embedding: Optional[ndarray]` (512-D ArcFace; for matching)
  - `salience_score: float` (precomputed base score)
  - `permissions: set[str]` (view/use scopes; supports per-surface rules)
  - `source: Literal["transcript", "manual", "integration"]`
  - `ttl: Optional[timedelta]` (expiry for sensitive memories)
  - `metadata: dict[str, Any]` (extensible key/value, audited)

- **Person**
  - `id: UUID`
  - `display_name: str`
  - `known_face_embeddings: list[ndarray]`
  - `preferences: dict[str, Any]`
  - `last_interaction_at: datetime`
  - `relationships: dict[str, float]` (closeness/trust)

- **InteractionContext** (per session/request)
  - `face_embedding: Optional[ndarray]`
  - `conversation_topics: list[str]`
  - `location: Optional[str]`
  - `emotion_signals: list[str]`
  - `device_state: dict[str, Any]` (battery, connectivity)

## 6. Retrieval Pipeline (v1 draft)
1. **Input normalization:** Build `ContextInput` DTO combining face embedding, transcript topics, location, and optional physiological signals.
2. **Identity resolution:**
   - Run cosine similarity between `ContextInput.face_embedding` and `Person.known_face_embeddings` (CosFace margin).
   - If unknown or low confidence, maintain provisional identity and avoid personal overlays.
3. **Candidate fetch:**
   - Query vector store for memories using semantic embedding of current transcript/topics.
   - Apply filters: permissions, TTL validity, location relevance, device safety (e.g., driving → restrict overlays).
4. **Scoring & ranking:**
   - Base score: similarity between memory.embedding and current context embedding.
   - Adjustments: salience decay (time-weighted), relationship weighting, emotion congruence, recency bias, diversity penalty for redundant cues.
   - Target latency: <50 ms with warm vector index and in-process re-ranker.
5. **Summarization & selection:**
   - Compress top-k memories into single-line cues (name + last interaction + key preference).
   - Enforce display budget (1–2 cues max) and safety constraints.
6. **Caching & feedback:**
   - Cache recent embeddings and last-surfaced cues per identity.
   - Collect lightweight feedback signals (dismiss/confirm) to update salience and model weights offline.

## 7. Open Questions / Next Steps
- Privacy model: what encryption strategy and key management are required for on-device vs cloud?
- Data retention: TTL defaults per source and jurisdictional compliance requirements.
- Edge vs cloud: when to store/cache embeddings locally vs remote vector DB?
- UX safety: how to detect unsafe contexts (e.g., driving) and suppress overlays?
- Evaluation: target metrics for recognition accuracy, retrieval precision, and latency budgets.
