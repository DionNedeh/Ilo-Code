# Ilo v11.0.0 → v11.5.0 Engineering Roadmap

**Status:** Plan of record — analysis only, no code changed.
**Author:** Prepared from a full line-by-line read of the v11.0.0 tree (`main.py`, `statera/*`, `frontend/src/*`) plus empirical profiling.
**Scope:** Merges *Improvement Suggestions 1* and *Improvement Suggestions 2* into one deduplicated, traceable backlog. Every sub-point from both source documents is mapped to exactly one roadmap item (see the traceability column in §3). Nothing is dropped.

> **How to read this doc.** §1 is the executive summary. §2 is the empirical validation the brief asked for (I profiled the RAG scan and measured the prompt-ordering token waste on this machine — real numbers, reproducible scripts). §3 is the master effort/impact/risk table with source traceability. §4 is the release sequencing across v11.0.0→v11.5.0. §5 are the hard invariants every item must preserve. §6 is the decision register (the "don't touch a line until you choose a direction" calls). §7 (Track A) and §8 (Track B) are the per-item deep dives — each with **What / Why / Where (exact files + current line numbers) / Diff sketch / Effort·Impact·Risk / Acceptance criteria / Test plan**. §9 is the file-by-file change index.

> **A note on line numbers.** The two source documents cite line numbers from an *earlier* tree; the code has since drifted (e.g. `_top_semantic` is now `documents.py:363`, not `:325`). **Every line number in this roadmap was re-verified against the current v11.0.0 working tree** and is annotated where the source's number was stale. Re-verify again at implementation time — this file is a map, not a lock.

---

## 1. Executive summary

Ilo v11.0.0 is a mature local-first agentic desktop workbench: a pywebview/WebView2 shell (`main.py`) over a FastAPI server (`statera/app.py`, **81 routes**) backed by WAL SQLite (`statera/db.py`, single-writer `RLock` + per-thread read-only connections). The agent core is `SessionRuntime` (`statera/runtime.py`, 2 757 lines): a plan→act→verify loop with a streaming OpenAI-compatible client, **23 tools**, a completion judge, run reflection into a review inbox, hybrid RAG (FTS5 + in-Python cosine), model routing (`primary`/`planner`/`coder`/`json`/`vision`/`embedding` — already partially wired), a bundled llama.cpp manager, and voice. The security posture is genuinely strong: socket-level SSRF pinning (`tools.py:121`), workspace path-escape + symlink/junction guards (`tools.py:431`), env allow-listing (`tools.py:1493`), per-process API auth.

That maturity is exactly why the backlog below is worth doing: **the scaffolding for most of it already exists.** Model routing, the blackboard, run steps, project plans, `NativeRuntimeSettings.speculative_decoding_enabled`/`draft_model_path`, the `experimental.multi_agent_enabled` flag, an `/api/experimental/orchestrator/runs` endpoint, an `/api/eval-lab/run` endpoint, and a `project_index`/`find_symbol`/`analyze_impact` code index are all present but **stubbed, disabled, or byte-for-byte re-computed every turn.** The work is mostly *activation and sharpening*, not green-field.

Two headline efficiency claims from the source docs were **empirically validated on this machine** (§2):

* **RAG scan:** the current per-turn pure-Python cosine + full SQLite BLOB re-decode costs **~100 ms at 2 000 chunks and ~360 ms at 5 000 chunks (1024-dim)**; a cached NumPy matrix path does the same scoring in **~1–2 ms** — a **70×–550× per-turn speedup**. NumPy 2.5.0 is *already installed* in the venv.
* **Prompt ordering:** **~340–900 tokens of byte-identical static contract text are re-sent every turn** and cannot be prefix-cached today because volatile memory/RAG/skills sit in front of them. Over a 10-turn run that is **~3 400–7 000 re-sent input tokens** a stable-prefix cache would cover.

The roadmap is split into **Track A** (14 incremental, high-confidence items — performance, cost, robustness) and **Track B** (11 radical, higher-ceiling items — capability leaps). The single hard rule threaded through all of it: **no regression in the security boundary** (§5). Broadening autonomy (B2, B10) is the whole point of those items, so their acceptance criteria treat the SSRF/path/allow-list invariants as non-negotiable gates.

---

## 2. Empirical validation (the brief's third deliverable)

Two reproducible micro-benchmarks were run against the actual algorithms in the tree. Scripts are in the scratchpad and reproduced in §9.3.

### 2.1 RAG semantic-scan profile

Reproduces `HybridRAG._dot`/`_top_semantic` (`documents.py:357–375`, the exact pure-Python generator + `heapq.nlargest`) and the per-turn BLOB decode that `db.document_chunks` (`db.py:2483`) performs on **every** `retrieve()` call, versus a cached `float32` NumPy matrix scored with one `matrix @ query` + `argpartition`.

Environment: Python 3.12.0, NumPy 2.5.0, Windows 11. Median of 5 repeats.

| chunks | dim | BLOB-decode / turn | py-scan / turn | **current per-turn (decode+scan)** | NumPy build (cache-miss) | **NumPy score (cache-hit)** | per-turn speedup |
|-------:|----:|-------------------:|---------------:|-----------------------------------:|-------------------------:|----------------------------:|-----------------:|
| 500    | 768 | 8.9 ms  | 14.9 ms  | **23.8 ms**  | 1.5 ms  | **0.70 ms** | **34×** |
| 2 000  | 768 | 40.5 ms | 59.7 ms  | **100.2 ms** | 4.3 ms  | **1.38 ms** | **73×** |
| 5 000  | 768 | 120.5 ms| 153.8 ms | **274.3 ms** | 7.4 ms  | **1.89 ms** | **145×** |
| 10 000 | 768 | 249.6 ms| 309.2 ms | **558.7 ms** | 16.9 ms | **1.29 ms** | **433×** |
| 2 000  | 1024| 53.2 ms | 81.0 ms  | **134.2 ms** | 3.7 ms  | **0.35 ms** | **388×** |
| 5 000  | 1024| 151.7 ms| 205.9 ms | **357.7 ms** | 8.8 ms  | **0.71 ms** | **504×** |

**Reading it.** The source docs claimed "10–100× on the scoring step." That is *conservative*. The scan alone is ~60 ms at 2k/768; the true per-turn cost also includes the ~40 ms full-table BLOB re-decode `retrieve()` repeats every message. The cached NumPy path collapses both to **1–2 ms**. Even the *cache-miss* rebuild (decode all BLOBs into one matrix) is 4–17 ms — an order of magnitude below today's steady state. This is the single cheapest win in the doc and directly validates **A1**.

### 2.2 Prompt-ordering token-waste measurement

`_build_messages` (`runtime.py:1427–1569`) appends the volatile sections first (agent prompt → memory → skills → compacted summary → RAG → project plan → uploaded-doc catalog) and the **static contract blocks last** (workspace contract, command contract, visual contract, run discipline, web-research discipline, thread discipline). Verbatim copies of those static strings were measured:

| static block | chars | ~tokens (char/4) | ~tokens (word) |
|---|---:|---:|---:|
| workspace_contract | 696 | 174 | 126 |
| command_contract | 431 | 108 | 87 |
| visual_contract | 1 388 | 347 | 273 |
| run_discipline | 165 | 41 | 32 |
| web_discipline | 662 | 166 | 127 |
| thread_discipline (static part) | 295 | 74 | 54 |
| **total static suffix** | **3 637** | **909** | **699** |

* Always-on (command tools off): **~340 tokens/turn** (word est.) of re-sent, uncacheable static text.
* Command + visual tools enabled: **~700–900 tokens/turn**.
* Over a 10-turn run: **~3 400–7 000 input tokens** re-sent that a stable-prefix cache would cover — *plus* the agent system prompt and the fixed tool schemas, which are also stranded behind the volatile prefix today.

**Reading it.** The waste is not just these ~900 tokens; it is that putting volatile content first makes the **entire** system block (system prompt + contracts + the 23 tool JSON schemas re-sent every turn in `stream_chat`) un-cacheable on OpenAI/Anthropic/Gemini. Reordering to a stable prefix (**A2**) turns all of it into cache hits. This validates the source docs' "single biggest cost/latency win" framing for cloud providers.

---

## 3. Master effort / impact / risk table

Effort: **S** ≤1 day · **M** ~2–4 days · **L** ~1–2 weeks · **XL** >2 weeks.
Impact and Risk: L/M/H. "Risk" = blast radius on correctness/security if done wrong.

### Track A — Performance, cost, robustness (incremental)

| ID | Item | Effort | Impact | Risk | Merges (S1 / S2) |
|----|------|:------:|:------:|:----:|------------------|
| **A1** | Vectorized RAG scoring + cached matrix + query-embedding LRU | S–M | **H** | L | S1-1.2 / S2-1.1 |
| **A2** | Prompt-cache-friendly system-prompt reorder + `cache_control` | M | **H** | M | S1-1.3 / S2-2.1 |
| **A3** | Token-accurate budgeting (tokenizer abstraction → `ContextTracker`) | M | M–H | L | S1-1.4 / S2-1.7 (counting) |
| **A4** | Parallel read-only tool execution | M | **H** | M | S1-1.1 / S2-1.4 |
| **A5** | Model-call retry/backoff (respect `Retry-After`) | S–M | M–H | L | — / S2-1.3 |
| **A6** | Read-only tool-result cache (files by mtime, web by TTL, research per-source) | M | M | M | S1-1.7 / S2-1.8 (research cache) |
| **A7** | Consolidate startup migration/backup gates (version→handler table) | S | L–M | L | S1-1.8 / — |
| **A8** | Collapse end-of-run calls (reflection∥memory, gate judge) | M | M | M | S1-1.5 / — |
| **A9** | Model-driven intent routing (retire brittle regex heuristics) | M | M | M | S1-1.6 / S2-1.5 + S2-2.4 |
| **A10** | Provider structured outputs (JSON schema) w/ lenient fallback | M | M | L | — / S2-2.7 |
| **A11** | Hybrid semantic memory + skills retrieval | M | M | L | — / S2-1.2 |
| **A12** | Pluggable search backend + readability extractor | M | M | M | — / S2-1.6 |
| **A13** | Hierarchical/rolling token-aware auto-compaction | M | M | M | — / S2-1.7 |
| **A14** | Windowed `read_file` (offset/line-range) + loop-nesting docs | S | M | L | — / S2-1.8 |

### Track B — Capability leaps (radical)

| ID | Item | Effort | Impact | Risk | Merges (S1 / S2) |
|----|------|:------:|:------:|:----:|------------------|
| **B1** | Tiered model routing for plumbing calls (judge/classify/memory/compact) | M | **H** | M | — / S2-2.2 |
| **B2** | Real multi-agent orchestration (planner → parallel workers → verifier) | XL | **VH** | **H** | S1-2.1 / S2-2.3 |
| **B3** | ANN vector store (sqlite-vec) behind `HybridRAG` | L | **H** | M | S1-2.2 / S2-2.5 (ANN) |
| **B4** | Code-aware workspace intelligence (tree-sitter symbol index) | L | H | M | — / S2-2.5 (code) |
| **B5** | MCP client support (consume external MCP servers) | L | **H** | **H** | — / S2-2.6 |
| **B6** | Measured self-improvement eval harness (A/B before adopt) | L–XL | H | M | S1-2.3 / S2-2.8 |
| **B7** | AST edits + auto-verify edit→test→fix loop (+ real unified-diff apply) | L–XL | H | M | S1-2.4 / S2-2.9 |
| **B8** | Structured memory graph (entities/relations) | L | M–H | M | S1-2.5 / — |
| **B9** | Speculative decoding + KV-cache reuse (native runtime) | M | H (local) | M | S1-2.6 / — |
| **B10** | Sandboxed command execution (Job Objects / container) | XL | **H** | **H** | S1-2.7 / S2-2.10 |
| **B11** | Frontend decomposition + transcript virtualization | L–XL | M→H | L | S1-2.8 / — |

**Traceability check:** S1 items 1.1–1.8, 2.1–2.8 → all mapped. S2 items 1.1–1.8, 2.1–2.10 → all mapped. S2-1.8 is a compound note (nested-loop docs, `read_file` windowing, research per-source cache) split across **A14** (docs + windowing) and **A6** (research cache). The S2 closing "direction decisions" (MCP, sandbox, second provider, ANN dep) become the **Decision Register (§6)**.

---

## 4. Release sequencing (v11.0.0 → v11.5.0)

Ordered by payoff-per-unit-risk, honoring the dependency edges (A2→B1, A3→A13, A1→B3, B4→B2, B1→B2, A9→B1/B2, A10→B6). Each release is independently shippable and leaves the security boundary intact.

**v11.1.0 — "Quiet speedups" (no behavior change users can see except speed/cost).**
A1 (vector RAG), A2 (prompt-cache reorder), A5 (retry/backoff), A7 (startup gates), A6 (tool-result cache). Low risk, immediately felt. A2 unblocks B1.

**v11.2.0 — "Correctness & economy."**
A3 (real tokens), A10 (structured outputs), A13 (hierarchical compaction, depends A3), A4 (parallel tools), B1 (tiered routing, depends A2+A10). This is where cloud cost drops hard.

**v11.3.0 — "Smarter retrieval & intent."**
A11 (hybrid memory), A9 (model-driven intent), A12 (pluggable search), A14 (windowed read + loop docs), B3 (ANN store, depends A1). Retrieval scales from "a few docs" to "a real corpus."

**v11.4.0 — "The capability leap."**
B4 (code index), B2 (multi-agent, depends B1+B4+A9), B7 (AST + verify loop). Ilo becomes a team, and coding tasks gain a self-correction cycle.

**v11.5.0 — "Platform & compounding quality."**
B6 (eval harness, depends A10), B5 (MCP client), B9 (speculative decoding), B10 (sandbox), B8 (memory graph, depends B3), B11 (frontend decomposition). Finishes the two most ambitious half-built ideas (eval + orchestration) and unlocks safe hands-off autonomy.

> **Guardrail:** B6 (measured eval) should land no later than the *start* of v11.4.0 so that B2/B7 changes to the agent itself are regression-protected. If schedule slips, pull B6 forward ahead of B2.

---

## 5. Cross-cutting invariants (hard acceptance gates for every item)

These are pass/fail gates on **every** PR in this roadmap. A change that touches these areas must prove non-regression, not merely "look fine."

1. **Security boundary is non-negotiable.** SSRF pinning (`tools.py:121 _PinnedNetworkBackend`, `_PinnedHTTPTransport:191`), workspace path/symlink/junction guards (`_resolve_workspace_path` `tools.py:431`, `_revalidate_workspace_path:463`), env allow-list (`_command_environment:1493`), credential/device-path rejection (`_validate_url:912`), and secret redaction in memory (`memory.py:_SENSITIVE_PATTERNS`) must remain intact. New network egress (A12, B5) routes through the pinned client or an explicitly-reviewed equivalent. New executors (B10) *tighten* the boundary, never loosen it.
2. **Off-by-default philosophy holds.** Command tools, visual verification, multi-agent, eval-lab, model-routing, speculative decoding, MCP, and sandbox executors stay behind `experimental.*`/`commands.*`/`native_runtime.*` flags until proven. No item silently raises default autonomy.
3. **Local-model compatibility preserved.** LM Studio/llama.cpp/Ollama-style servers reject object `tool_choice` and JSON-schema `response_format`; every provider-specific feature (A2 `cache_control`, A10 structured outputs, A4 parallel calls) must degrade to the current string/lenient path via capability probing (`llm.py:capability_profile:66`).
4. **Transcript determinism.** Parallelism (A4) and concurrency (A8, B2) must append messages/tool-results in a deterministic, original-call order so replays and reflection bundles stay stable.
5. **Test suite stays green + grows.** All **156** existing tests (`pytest --collect-only` → 156) pass unchanged unless a behavior deliberately changes (then the test is updated in the same PR with rationale). Each item adds tests (see per-item "Test plan"). New deps get a `requirements.txt` pin.
6. **Graceful degradation.** Every new external dependency (numpy already present; `tiktoken`, `sqlite-vec`, `tree-sitter`, MCP SDK) must have a feature-detect + fallback so a packaging miss degrades to today's behavior instead of crashing (mirrors the existing PyMuPDF→pypdf fallback in `documents.py:108`).

---

## 6. Decision register (choose before writing a line)

The source docs correctly flag that several items are *direction* decisions, not just code. Resolve these with the product owner up front; each has a recommended default.

| # | Decision | Options | Recommended default | Blocks |
|---|----------|---------|---------------------|--------|
| D1 | Token counter dependency | `tiktoken` (OpenAI BPE, ~5 MB) vs. pure-Python heuristic only | **Ship heuristic always; load `tiktoken` opportunistically** (feature-detect) so packaging never hard-depends on it | A3, A13 |
| D2 | ANN backend | `sqlite-vec` (stays single-file, zero-server) vs. `hnswlib`/FAISS (faster, in-memory rebuild) | **`sqlite-vec`** — preserves the single-file SQLite ethos and PyInstaller story | B3, B8 |
| D3 | Code index parser | `tree-sitter` (accurate, per-language grammars, native wheels) vs. regex/ctags | **`tree-sitter` for top 5 langs, regex fallback** (extends existing `project_index.py`) | B4, B7 |
| D4 | Second/cheap model source | reuse native llama.cpp slot vs. require a configured cloud "secondary" | **Reuse existing `json`/`planner` route profiles** (already in schema) — no new provider required | B1 |
| D5 | Search API | Brave / Tavily / SearXNG (key required) vs. scraper-only | **Pluggable: use API when key present, keep scraper as no-key fallback** | A12 |
| D6 | MCP transport surface | stdio-only vs. stdio + HTTP/SSE | **stdio first** (local servers), gate remote transports behind SSRF review | B5 |
| D7 | Sandbox mechanism | Windows Job Objects + AppContainer vs. WSL/container executor | **Job Object + restricted token first** (no WSL dependency), container as opt-in | B10 |
| D8 | Frontend state lib | Zustand vs. Jotai vs. hand-rolled | **Zustand** (smallest mental overhead over the existing SSE reducer) | B11 |

---

# Track A — Deep dives (incremental)

---

## A1 — Vectorized RAG scoring + cached matrix + query-embedding LRU

**Merges:** S1-1.2, S2-1.1. **Effort:** S–M · **Impact:** H · **Risk:** L. **Empirically validated: §2.1 (34×–550× per-turn).**

### What
Replace the per-turn pure-Python cosine scan and full-table BLOB re-decode with (a) a per-session, memory-cached, normalized `float32` NumPy matrix keyed by a document-set signature, scored with a single `matrix @ query`, and (b) an LRU cache on `query text → embedding vector`.

### Why
`HybridRAG.retrieve` (`documents.py:239`) calls `db.document_chunks(session_id, embedding_model)` (`documents.py:264`) on **every** turn, which re-runs `SELECT document_chunks.*` and decodes *every* embedding BLOB via `_decode_row` (`db.py:1176`), then `_top_semantic` (`documents.py:363`) scores each with the interpreted `_dot` generator (`documents.py:357`). §2.1 measures this at ~100 ms (2k chunks) to ~360 ms (5k, 1024-dim) per message — the dominant cost of the "context" phase, repeated every turn. Embeddings are **already normalized at write time** (`db.py:_encode_embedding:112` → `_normalize_embedding`), so a cached matrix needs no re-normalization; scoring is a pure dot product.

### Where (verified)
* `statera/documents.py:357` `_dot`, `:363` `_top_semantic`, `:239–285` `retrieve`, `:264` the per-turn reload.
* `statera/db.py:2483` `document_chunks`, `:112` `_encode_embedding` (normalizes), `:119` `_decode_embedding`.
* Cache lifecycle hook: `retrieve` is the only hot caller; invalidation signal is the set of `(chunk_id, embedding_model)` for the session — cheaply summarized by `db.document_chunk_embedding_counts` (`db.py:2504`) + a max-`updated_at`.

### Diff sketch
```python
# documents.py — new: a per-session cached matrix, built once, scored with numpy.
try:
    import numpy as _np
    _HAVE_NUMPY = True
except Exception:            # packaging safety — degrade to the pure-Python path
    _HAVE_NUMPY = False

class _SemanticIndexCache:
    """Per-session normalized float32 matrix, keyed by (model, chunk_count, max_updated_at)."""
    def __init__(self) -> None:
        self._entries: dict[str, tuple[tuple, "np.ndarray", list[dict]]] = {}
        self._lock = threading.Lock()

    def get_or_build(self, session_id, model, signature, candidates):
        with self._lock:
            cached = self._entries.get(session_id)
            if cached and cached[0] == signature:
                return cached[1], cached[2]
        # Embeddings are already L2-normalized at ingest (db._encode_embedding), so stack as-is.
        mat = _np.asarray([c["embedding"] for c in candidates], dtype="<f4")
        with self._lock:
            self._entries[session_id] = (signature, mat, candidates)
        return mat, candidates

# retrieve(): replace the scan block
if embedding_model and _HAVE_NUMPY:
    counts = await asyncio.to_thread(self.db.document_chunk_embedding_counts, session_id, embedding_model)
    signature = (embedding_model, counts["embedded"], await asyncio.to_thread(self.db.documents_signature, session_id))
    query_vector = await self._embed_query_cached(query, embedding_model, model_client)   # LRU
    candidates = await asyncio.to_thread(self.db.document_chunks, session_id, embedding_model)
    mat, rows = self._semantic_cache.get_or_build(session_id, embedding_model, signature, candidates)
    q = _np.asarray(self._normalize_vector(query_vector), dtype="<f4")
    scores = mat @ q
    top = _np.argpartition(-scores, min(20, len(scores) - 1))[:20]
    semantic = [rows[i] for i in top[_np.argsort(-scores[top])]]
```
Add `db.documents_signature(session_id)` = `SELECT COUNT(*), MAX(updated_at) FROM document_chunks WHERE session_id=?`. Keep `_top_semantic`/`_dot` as the `not _HAVE_NUMPY` fallback. RRF fusion (`documents.py:268–276`) is **unchanged** — it still consumes ranked `semantic`/`lexical` lists.

### Acceptance criteria
* `retrieve()` returns **identical top-k ordering** to the pure-Python path on a fixed corpus (golden test; ties broken identically by index).
* Cache invalidates when a document is added/removed/re-embedded (add/delete a doc mid-session → next `retrieve` rebuilds).
* No numpy → falls back with no error; parity test runs both paths.
* Measured ≥20× per-turn scan speedup at ≥2 000 chunks in a benchmark test.

### Test plan
Extend `tests/test_rag_v7.py`: (1) parity test numpy vs. pure-Python on a seeded 300-chunk corpus; (2) invalidation test; (3) `monkeypatch` `_HAVE_NUMPY=False` fallback test; (4) micro-benchmark asserting the speedup floor.

---

## A2 — Prompt-cache-friendly system-prompt reorder + `cache_control`

**Merges:** S1-1.3, S2-2.1. **Effort:** M · **Impact:** H · **Risk:** M. **Empirically validated: §2.2 (~340–900 tokens/turn re-sent).**

### What
Invert `_build_messages` so the **invariant** blocks form a stable prefix (agent system prompt → workspace/command/visual/run/web/thread contracts → tool schemas) and the **volatile** blocks (memory, skills, compacted summary, RAG, project plan, uploaded-doc catalog, current objective) form a clearly delimited suffix. For Anthropic, attach `cache_control: {"type":"ephemeral"}` breakpoints at the end of the stable prefix; for OpenAI/Gemini the stable prefix caches automatically.

### Why
§2.2 shows the static contracts (~340–900 tokens) are re-sent verbatim every turn but sit *after* volatile content, so no provider prefix-cache can cover them — nor the agent system prompt nor the 23 re-sent tool schemas. On cloud providers this is the single biggest cost/latency lever (cached input tokens are ~10× cheaper and cut time-to-first-token). Pure efficiency, zero capability change.

### Where (verified)
* `statera/runtime.py:1427–1569` `_build_messages`. Today the order is: `system = agent.system_prompt` (`:1450`) → memory (`:1451`) → skills (`:1453`) → compacted summary (`:1455`) → RAG (`:1457`) → project plan (`:1463`) → uploaded-doc catalog (`:1469`) → **workspace contract (`:1487`)** → command contract (`:1499`) → run discipline (`:1531`) → web discipline (`:1535`) → thread discipline (`:1547`).
* `statera/llm.py:128 _completion_payload`, `:326 stream_chat` (where `tools`/messages become the request body), `:66 capability_profile` (provider gate).

### Diff sketch
```python
# runtime.py _build_messages — assemble two parts, stable first.
stable: list[str] = [agent.get("system_prompt") or "You are Ilo."]
stable.append(WORKSPACE_CONTRACT.format(project_root=project_root))   # was appended last
if app_settings.commands.enabled:
    stable.append(COMMAND_CONTRACT.format(...))
    if app_settings.commands.visual_verification_enabled:
        stable.append(VISUAL_CONTRACT)
stable.append(RUN_DISCIPLINE); stable.append(WEB_DISCIPLINE); stable.append(THREAD_DISCIPLINE_STATIC)

volatile: list[str] = []
if memory_context: volatile.append("Long-term memory (use only when relevant):\n" + memory_context)
if skill_context:  volatile.append("Relevant approved skills:\n" + skill_context)
if session.get("context_summary"): volatile.append("Compacted prior thread context:\n" + ...)
if rag_context:    volatile.append(RAG_PREAMBLE + rag_context)
if project_plan_context: volatile.append(project_plan_context + PLAN_REPORTING_SUFFIX)
if uploaded_documents:   volatile.append(UPLOADED_DOC_CONTRACT.format(catalog=catalog))
volatile.append(f"Current turn request: {current_objective}")
if conversational_turn:  volatile.append(CONVERSATIONAL_SUFFIX)

messages = self._compose_system(stable, volatile, capability_profile)   # provider-aware
```
```python
# _compose_system — Anthropic gets a cache breakpoint; others get plain concatenation.
def _compose_system(self, stable, volatile, profile):
    stable_text = "\n\n".join(stable)
    volatile_text = "\n\n".join(volatile)
    if profile.get("provider") == "anthropic" or "anthropic" in profile.get("host", ""):
        return [{"role": "system", "content": [
            {"type": "text", "text": stable_text, "cache_control": {"type": "ephemeral"}},
            {"type": "text", "text": volatile_text},
        ]}]
    return [{"role": "system", "content": stable_text + "\n\n" + volatile_text}]
```
> **Confirm against current API docs before wiring provider fields** (the source doc flags this correctly — cache-control block shape and minimum cacheable-prefix length differ by provider). The Anthropic block-list system format must be gated by capability probing so local servers still receive a plain string.

### Acceptance criteria
* Byte-identical *stable* prefix across consecutive turns of a run (unit test snapshots the prefix twice and asserts equality).
* Volatile content still present and correctly delimited (RAG/memory retrieval tests unchanged).
* Local/OpenAI path emits a plain-string system message (no block list); only Anthropic path emits `cache_control`.
* On Anthropic, a 5-turn run reports `cache_read_input_tokens > 0` after turn 1 (integration smoke, mocked).

### Test plan
New `tests/test_prompt_cache_v11.py`: prefix-stability snapshot; provider-matrix test (local→string, anthropic→blocks); ensure `current_objective` and RAG land in the suffix. Regression-run `test_context_tracking.py`.

---

## A3 — Token-accurate budgeting

**Merges:** S1-1.4, S2-1.7 (counting half). **Effort:** M · **Impact:** M–H · **Risk:** L. **Decision:** D1.

### What
Introduce a small `TokenCounter` abstraction (tiktoken for OpenAI-family when available; provider `count_tokens` where cheap; a calibrated char/word heuristic fallback for GGUF/local) and route all budgeting through it instead of raw character math.

### Why
Budgeting is character-based throughout: `_history_limit_for_context` uses `budget // 1500` (`runtime.py:1579`); `_manual_compaction_source_chars` (`:1582`), `_reflection_source_chars` (`:1589`), and the compaction/reflection fitters (`_fit_reflection_bundle:1219`) all use char caps. Chars-per-token varies 2–5× (code vs. prose vs. CJK), so today's budgeting both under-fills context (wasted capability) and risks overflow on dense inputs.

### Where (verified)
* `statera/runtime.py:1572 _history_limit_for_context`, `:1582`, `:1589`, `:1219 _fit_reflection_bundle`, `:1612 _compact_session_context`.
* `statera/context.py` (64 lines) `ContextTracker` — the natural home/route for estimates.
* `statera/schemas.py:33 context_budget` (already tokens-semantics in name; today divided by a magic 1500 to fake tokens).

### Diff sketch
```python
# statera/tokens.py (new)
class TokenCounter:
    def __init__(self, model: str, provider: str):
        self._enc = None
        if provider in {"openai", "custom"}:
            try:
                import tiktoken
                self._enc = tiktoken.encoding_for_model(model) if _known(model) else tiktoken.get_encoding("o200k_base")
            except Exception:
                self._enc = None
    def count(self, text: str) -> int:
        if self._enc is not None:
            return len(self._enc.encode(text))
        # Calibrated heuristic: max of word-ish and char/3.6 (empirically closer than char/4 for code).
        words = len(_TOKEN_RE.findall(text))
        return max(words, round(len(text) / 3.6))
```
```python
# runtime.py — route char caps through tokens (keep char clamps as hard safety rails).
def _history_limit_for_context(settings, counter) -> int:
    explicit = int(settings.model.history_message_limit or 0)
    if explicit > 0: return max(1, min(400, explicit))
    return max(24, min(120, settings.model.context_budget // 900))  # token budget, not chars//1500
```
`ContextTracker` (`context.py`) gains `counter` and reports token estimates so `context_usage` (`runtime.py:110`) shows real numbers in the UI.

### Acceptance criteria
* With tiktoken absent, heuristic is within ±15 % of tiktoken on a mixed prose/code corpus (unit test with a bundled fixture of pre-counted strings).
* No context-overflow ModelError on a dense-code fixture that overflows today.
* `context_usage` surfaces token estimates; existing `test_context_tracking.py` updated for token semantics.

### Test plan
`tests/test_tokens_v11.py`: heuristic-vs-tiktoken accuracy band; CJK/code/prose fixtures; overflow-avoidance test. Update `test_context_tracking.py`.

---

## A4 — Parallel read-only tool execution

**Merges:** S1-1.1, S2-1.4. **Effort:** M · **Impact:** H · **Risk:** M.

### What
When the model emits multiple tool calls in one step, run the **read-only** subset concurrently (`asyncio.gather` over `asyncio.to_thread`) while keeping **mutating** tools strictly serial and ordered. Append tool-result messages back in original call order.

### Why
`_run_model_loop` executes `for call in parsed_calls:` (`runtime.py:1974`) and awaits each `self.tools.execute` (`:2096`) sequentially. A turn that reads 4 files or runs 3 web searches pays the *sum* of latencies; read-only tools have no ordering dependency. The tool manifest already labels access (`tool_manifest.py:14 access="read-only"`), and the duplicate-mutation guard (`runtime.py:2087`) already assumes serialized writes.

### Where (verified)
* `statera/runtime.py:1974` the sequential loop, `:2096` `execute`, `:2087` mutation dedup, `:2442 _is_mutating_tool` (the authoritative mutating set).
* `statera/tools.py:341 execute` is thread-safe per call (opens its own settings/DB handles).
* Read-only set (complement of `_is_mutating_tool`): `read_file, read_document, list_files, search_files, workspace_map, find_symbol, analyze_impact, search_knowledge, read_knowledge_source, web_search, read_url, web_fetch, research_web, propose_patch, capture_screen`(?—capture is side-effecting, treat as serial).

### Diff sketch
```python
# runtime.py — partition, run read-only concurrently, keep the transcript deterministic.
readonly, mutating = [], []
for call in parsed_calls:
    (mutating if self._is_mutating_tool(call["function"]["name"]) else readonly).append(call)

async def _run_one(call):
    # (arg-parse + missing-required validation identical to today, refactored into a helper)
    return call, await self._execute_validated(session_id, call, assistant_message_id, completed_mutations)

results: dict[str, ToolResult] = {}
if readonly:
    for call, res in await asyncio.gather(*(_run_one(c) for c in readonly)):
        results[call["id"]] = res
for call in mutating:                      # serial; preserves dedup + write ordering
    _, res = await _run_one(call)
    results[call["id"]] = res

for call in parsed_calls:                   # emit thread items + tool messages in ORIGINAL order
    await self._emit_tool_result(session_id, call, results[call["id"]], ...)
```
Refactor the per-call body (arg parse `:1978`, missing-required `:1990`, dedup `:2087`, execute `:2096`, blackboard `:2162`, tool message `:2176`) into `_execute_validated` + `_emit_tool_result` so both paths share one code path. Concurrency cap: `asyncio.Semaphore(min(4, len(readonly)))` to bound the thread pool.

### Acceptance criteria
* Multi-read turn transcript is **identical order** to serial execution (messages/tool items appended by original index).
* Mutating tools never overlap; dedup guard still blocks duplicate writes within a turn.
* A 3-file-read turn completes in ~max(latency) not ~sum (timing test with a slow fake tool).
* `capture_screen`/`run_command`/all `_is_mutating_tool` names stay serial.

### Test plan
`tests/test_parallel_tools_v11.py`: order-determinism test with instrumented fake tools; concurrency timing test; mutation-serialization test; failure-in-one-readonly doesn't abort siblings.

---

## A5 — Model-call retry/backoff

**Merges:** S2-1.3. **Effort:** S–M · **Impact:** M–H · **Risk:** L.

### What
Wrap `complete_json` and the *pre-stream* connection of `stream_chat` in bounded exponential backoff with jitter, retrying only safe/idempotent failures (HTTP 408/409/429/5xx, connect/read-timeout errors), honoring `Retry-After`. **Never** retry once streaming tokens have begun (mid-stream corruption is not idempotent).

### Why
`OpenAICompatibleClient.complete_json` (`llm.py:286`) and `stream_chat` (`llm.py:326`) make a single attempt each. A momentary 429/5xx or a socket hiccup — common against local servers stalling on cold model load — surfaces as `ModelError` and fails the whole turn. Bounded retry materially raises real-world completion rate.

### Where (verified)
* `statera/llm.py:286 complete_json`, `:326 stream_chat` (the `client.stream(...)` context at `:349`; retry must happen *before* the first `yield` at `:367`).
* Callers that benefit: judge (`runtime.py:2313`), reflection (`:1006`), compaction (`:1114`), memory (`memory.py:98`), and the act loop (`runtime.py:1732`).

### Diff sketch
```python
# llm.py
_RETRYABLE_STATUS = {408, 409, 425, 429, 500, 502, 503, 504}

async def _with_retry(self, make_request, *, attempts=3, base=0.5, cap=8.0):
    for attempt in range(attempts):
        try:
            return await make_request()
        except httpx.HTTPStatusError as exc:
            status = exc.response.status_code
            if status not in _RETRYABLE_STATUS or attempt == attempts - 1:
                raise
            delay = self._retry_after(exc.response) or min(cap, base * 2 ** attempt) + random.uniform(0, 0.3)
        except (httpx.ConnectError, httpx.ReadTimeout, httpx.RemoteProtocolError) as exc:
            if attempt == attempts - 1: raise
            delay = min(cap, base * 2 ** attempt) + random.uniform(0, 0.3)
        await asyncio.sleep(delay)
```
`complete_json` wraps its POST in `_with_retry`. For `stream_chat`, retry only the `client.stream(...)` open + first-error check (`:356 is_error`); once `aiter_lines` yields, no retry. Retries are configurable via a new `ModelSettings.max_request_retries` (default 3, `0` disables).

### Acceptance criteria
* A mocked 429-then-200 sequence succeeds transparently; `Retry-After` respected within ±20 %.
* A mid-stream disconnect after tokens are yielded does **not** retry (raises, as today).
* Non-retryable 400/401/404 fail immediately (no wasted backoff).
* `max_request_retries=0` reproduces today's single-attempt behavior exactly.

### Test plan
`tests/test_llm_retry_v11.py` with an httpx `MockTransport` scripting status sequences; assert attempt counts, delay honoring, and the no-mid-stream-retry invariant.

---

## A6 — Read-only tool-result cache

**Merges:** S1-1.7, S2-1.8 (research per-source cache). **Effort:** M · **Impact:** M · **Risk:** M.

### What
A small keyed cache inside `ToolRegistry`: `(tool, canonicalized_args) → ToolResult`. File reads invalidate by `mtime`; web reads (`read_url`/`web_fetch`/`web_search`/`research_web` sources) carry a TTL. Never cache mutating tools. Add a per-source cache inside `research_web` so its fan-out doesn't re-fetch pages already read this run.

### Why
No caching exists across turns; agents re-read the same file and re-run the same search constantly. `research_web` (`tools.py:944`) fans out reads with a 30 s wall-clock (`:971`) but has no per-source cache, so repeated research re-fetches. Cheap dedup cuts tokens, latency, and duplicate I/O. `mtime` tracking is already used for the image cache (`runtime.py:2607`), establishing the pattern.

### Where (verified)
* `statera/tools.py:341 execute` (wrap read-only dispatch), `:290 __init__` (add the cache + lock), `:472 _read_file` (mtime key), `:944 _research_web` / `:962 read_source` (per-source memo).
* Mutating set to exclude: `runtime.py:2442 _is_mutating_tool`.

### Diff sketch
```python
# tools.py __init__
self._result_cache: OrderedDict[str, tuple[float, ToolResult]] = OrderedDict()
self._result_cache_lock = threading.Lock()
_CACHEABLE = {"read_file","read_document","list_files","search_files","workspace_map",
              "find_symbol","analyze_impact","read_url","web_fetch","web_search","research_web"}
_WEB_TTL = 900.0

def _cache_key(self, session_id, tool, args):
    canon = json.dumps(args, sort_keys=True, separators=(",", ":"))
    stamp = ""
    if tool in {"read_file","read_document"}:
        try: stamp = str(self._resolve_workspace_path(session_id, args.get("path",""), self.db.get_settings()).stat().st_mtime_ns)
        except Exception: stamp = "0"
    return f"{session_id}|{tool}|{canon}|{stamp}"

# execute(): consult cache for _CACHEABLE, TTL for web tools, store result on success.
```
`research_web`'s `read_source` (`tools.py:962`) checks the same cache before `_read_url`, so overlapping sources across quick/deep calls hit warm.

### Acceptance criteria
* Second identical `read_file` in a run returns cached bytes; editing the file (mtime change) busts it.
* Web results expire after TTL; `web_search` for a new query is never served stale.
* Mutating tools are never cached (write→read-back returns fresh content).
* Cache is bounded (LRU cap) and cleared on `ToolRegistry.close`.

### Test plan
`tests/test_tool_cache_v11.py`: mtime invalidation; TTL expiry (monkeypatched clock); mutating-exclusion; bound/eviction; research per-source memo hit.

---

## A7 — Consolidate startup migration/backup gates

**Merges:** S1-1.8. **Effort:** S · **Impact:** L–M · **Risk:** L.

### What
Replace the 13 sequential `_backup_before_vN()` probes with a single schema-version read dispatched through a `version → handler` table, then run only the relevant backup/migration steps.

### Why
`Database.__init__` (`db.py:161–174`) calls `_backup_before_v6 … _backup_before_v18` (13 methods, `db.py:177–361`) plus `migrate()` on every launch. Each probe re-queries `sqlite_master` and `schema_meta` for the version (2 queries × 13 = 26 redundant reads) — pure desktop cold-start latency that grows every release. Behavior-preserving consolidation reads the version **once**.

### Where (verified)
* `statera/db.py:161–174` the call block, `:177–361` the 13 methods, `:363 _create_backup`, `:384 migrate`, `:394 schema_meta` table.

### Diff sketch
```python
# db.py __init__ — one version read, table-driven dispatch.
current = self._read_schema_version()          # single sqlite_master + schema_meta probe
_BACKUP_STEPS = [                              # (from_version, label)
    (5, "5.5-pre-v6"), (6, "6.0-pre-v7"), (7, "7.0-pre-v8"), (8, "8.0-pre-v9"),
    (9, "8.1-pre-v10"), (10, "8.8-pre-v11"), (11, "9.0-pre-v12"), (12, "9.0-pre-v13"),
    (13, "9.0-pre-v14"), (14, "9.0-pre-v15"), (15, "9.0-pre-v16"), (16, "10.0-pre-v17"),
    (17, "11.0-pre-v18"),
]
for from_version, label in _BACKUP_STEPS:
    if current == from_version:                # exactly one can match a given launch
        self._backup_once(f"{self.path.stem}-{label}{self.path.suffix}")
        break
self.migrate(); self.seed_defaults()
```
`_read_schema_version()` handles the legacy "has `memory_records` but no `schema_meta`" v5.5 case that `_backup_before_v6` special-cased. Each historical backup filename is preserved exactly (byte-compatible upgrade path for existing installs).

### Acceptance criteria
* Fresh DB, and DBs stamped at each of v5.5…v18, produce **identical** backup files + final schema as today (golden fixtures per version).
* Startup issues ≤2 schema probes instead of 26 (assert via a query counter).
* No backup created when already current.

### Test plan
`tests/test_migration_gates_v11.py`: seed a DB at each historical version, run init, assert backup filename set == legacy behavior and query count dropped.

---

## A8 — Collapse end-of-run model calls

**Merges:** S1-1.5. **Effort:** M · **Impact:** M · **Risk:** M.

### What
(a) Run reflection and memory consolidation **concurrently** (`asyncio.gather`) since neither depends on the other; (b) **gate the judge** — skip it when the last turn made no tool calls and produced a clean terminal answer; (c) optionally merge reflection + memory extraction into one structured call sharing the `_reflection_bundle`.

### Why
After the loop, a run fires up to three serial model calls on the critical path to "run done": memory consolidation (`runtime.py:787`), reflection (`:813`), and compaction (`:838`), plus the per-turn judge (`:690`). Memory and reflection both analyze the same transcript and neither depends on the other — today they run sequentially (`:787` then `:811`). The judge fires after every non-conversational turn even when the turn produced a terminal answer with no tools.

### Where (verified)
* `statera/runtime.py:767–808` memory block, `:811–836` reflection block, `:837–861` compaction, `:687–690` judge invocation, `:2300 _judge`, `:1148 _reflection_bundle` (shared artifact), `:975 reflect_run`.

### Diff sketch
```python
# runtime.py run_session — reflection ∥ memory instead of sequential.
async def _memory_task():
    if not _should_consolidate: return {"status": "skipped"}
    created = await self.memory.consolidate(session_id, model_client, 12, 2, final_history)
    return {"status": "completed", "candidates": len(created)}

async def _reflection_task():
    if not auto_reflect_enabled: return diagnostics.get("reflection")
    r = await self.reflect_run(run["id"], messages=final_history)
    return {"status": "completed", "candidates": sum(len(r[k]) for k in ("improvements","memories","skills"))}

diagnostics["memory"], diagnostics["reflection"] = await asyncio.gather(
    _memory_task(), _reflection_task(), return_exceptions=False)   # both read final_history; no write conflict
```
```python
# Judge gating — skip when the terminal turn was clean and tool-free.
skip_judge = (stats.get("tool_calls", 0) == 0 and content and not pending_prompt)
if conversational_turn or not settings.run_completion_judge_enabled or skip_judge:
    decision = {"verdict": "done", "reason": "Clean terminal answer; judge skipped."}
else:
    decision = await self._judge(objective, content, messages, agent, judge_client)
```
Both write different DB tables (memory_records vs improvements/skills) so concurrent execution is race-free; keep `update_run_diagnostics` writes ordered after the gather.

### Acceptance criteria
* Reflection + memory produce the same candidates as serial execution (order-independent).
* Judge is skipped only on tool-free clean terminal turns; still runs when tools were used or output is empty.
* No DB write races (memory and improvement rows both persist).
* Perceived run-completion latency drops on tool-heavy runs (timing test).

### Test plan
Extend `tests/test_runs_skills_improvements.py`: concurrent reflection/memory parity; judge-skip matrix (tools vs. no-tools × empty vs. non-empty); diagnostics ordering.

---

## A9 — Model-driven intent routing

**Merges:** S1-1.6, S2-1.5, S2-2.4. **Effort:** M · **Impact:** M · **Risk:** M. **Pairs with B1 (cheap model).**

### What
Retire the two load-bearing regex heuristics as *primary* deciders. Keep them as a fast path; on low-confidence, fold a single structured routing decision into the loop that returns `{turn_type, needs_tools, likely_writes, complexity}` and drives tool exposure, forced-write mode, judge on/off, and reflection on/off.

### Why
`_is_conversational_turn` (`runtime.py:2341`, 4 regex patterns, ≤80 chars) disables *all* tools/memory/reflection for the turn when it fires — and misses "thanks, now fix the bug", non-English, "ok do it". `_looks_like_file_write_request` (`runtime.py:2499`, keyword+extension matching) forces `write_file` via the `tool_choice=required` narrowing hack (`:1724`) — it misses "put that in app.py" / "jot this down as hi.py" and false-fires on prose containing "build". These brittle classifiers fork real behavior.

### Where (verified)
* `statera/runtime.py:2341 _is_conversational_turn` (used at `:470`, `:577`, `:687`, `:2455`), `:2499 _looks_like_file_write_request` (used at `:1674`), `:1724` forced-write narrowing.
* Natural home for the structured call: alongside `_judge` (`:2300`) using the `json` route client — pairs with **B1**.

### Diff sketch
```python
# runtime.py — fast path first, model classifier only when ambiguous.
def _fast_intent(text) -> dict | None:
    if self._is_conversational_turn(text):        # high-precision keep
        return {"turn_type": "conversational", "needs_tools": False, "likely_writes": False}
    if self._looks_like_file_write_request(text): # high-precision keep
        return {"turn_type": "task", "needs_tools": True, "likely_writes": True}
    return None                                   # ambiguous → ask the model

async def _classify_turn(self, objective, json_client) -> dict:
    fast = self._fast_intent(objective)
    if fast is not None: return {**fast, "source": "heuristic"}
    raw = await json_client.complete_json([{"role":"system","content":ROUTER_PROMPT},
                                           {"role":"user","content":objective[:2000]}], max_tokens=120)
    parsed = self._parse_json_object(raw, "turn routing")
    return {"turn_type": parsed.get("turn_type","task"), "needs_tools": bool(parsed.get("needs_tools", True)),
            "likely_writes": bool(parsed.get("likely_writes", False)),
            "complexity": parsed.get("complexity","medium"), "source": "model"}
```
The forced-write narrowing (`:1724`) stays, but is now gated on `intent["likely_writes"]` *and* only kept as a hard fallback for **known-weak local models** (probe via `capability_profile`). Router runs on the cheap `json` route so it adds negligible cost.

### Acceptance criteria
* All current `test_*` conversational/file-write behaviors still pass via the fast path (no model call for clear cases).
* Ambiguous paraphrases ("put that in app.py", "could you persist that", non-English "save this") route correctly in a labeled fixture set (≥90 % on 30 cases).
* Router failure falls back to today's heuristic verdict (never crashes the loop).
* Strong cloud models don't trigger the forced-write narrowing.

### Test plan
`tests/test_intent_routing_v11.py`: fast-path parity; ambiguous-case labeled set with a fake router model; fallback-on-router-failure; forced-write gating by capability profile.

---

## A10 — Provider structured outputs (JSON schema)

**Merges:** S2-2.7. **Effort:** M · **Impact:** M · **Risk:** L. **Enables B6.**

### What
When the provider advertises structured-output support, request JSON-schema-constrained responses for the judge, reflection, memory, compaction, and router calls. Keep the current fence-stripping lenient parser as the local-model fallback.

### Why
Judge (`runtime.py:2323`), reflection (`:1016`), memory (`memory.py:105`), compaction (`:1121`), and the plan/router calls all emit JSON that is fence-stripped and `json.loads`'d via `_parse_json_object` (`runtime.py:1414`) / `_parse_model_candidates` (`memory.py:153`). Providers with JSON-schema/structured outputs eliminate an entire class of parse failures (and reduce reliance on the `allow_tagged_tool_calls` fallback at `runtime.py:1833`). Capability probing already exists (`llm.py:66`).

### Where (verified)
* `statera/llm.py:128 _completion_payload`, `:286 complete_json`, `:66 capability_profile`.
* Consumers: `runtime.py:1414 _parse_json_object`, `:2300 _judge`, `:975 reflect_run`, `:1073 compact_session`; `memory.py:70 consolidate`.
* Prompts define the shapes already (`JUDGE_PROMPT`, `RUN_REFLECTION_PROMPT`, `COMPACTION_PROMPT`, `FACT_PROMPT`) — codify them as JSON Schemas.

### Diff sketch
```python
# llm.py — complete_json gains an optional schema.
async def complete_json(self, messages, max_tokens=400, *, schema: dict | None = None) -> str:
    payload = self._completion_payload(messages, max_tokens=max_tokens, stream=False, temperature=0.2)
    if schema and self._supports_structured_output():   # api.openai.com + gemini openai-compat
        payload["response_format"] = {"type": "json_schema",
                                      "json_schema": {"name": schema["title"], "schema": schema, "strict": True}}
    # ... unchanged POST + lenient extraction (still strips fences if a server ignores response_format)

def _supports_structured_output(self) -> bool:
    host = self._host()
    return host == "api.openai.com" or "generativelanguage.googleapis.com" in host
```
Define `statera/schemas_json.py` with `JUDGE_SCHEMA`, `REFLECTION_SCHEMA`, `MEMORY_SCHEMA`, `COMPACTION_SCHEMA`, `ROUTER_SCHEMA`. Pass the matching schema at each call site. The parser stays as-is (structured mode just makes it a no-op fast path).

### Acceptance criteria
* On a structured-capable mock provider, malformed-JSON injection tests can't reach the parser (schema-valid guaranteed).
* On a local/no-`response_format` mock, behavior is byte-identical to today (fence-strip path).
* No schema is sent to hosts that don't advertise support.

### Test plan
`tests/test_structured_outputs_v11.py`: schema-passed-only-when-supported; parity of parsed result vs. lenient path; fallback when server ignores `response_format`.

---

## A11 — Hybrid semantic memory + skills retrieval

**Merges:** S2-1.2. **Effort:** M · **Impact:** M · **Risk:** L. **Depends conceptually on A1's embed pipeline; benefits from B3.**

### What
Embed approved/pinned memories and skills with the existing embedding pipeline and fuse embedding similarity with the current FTS5 bm25 via the same reciprocal-rank fusion already written for documents.

### Why
`MemoryService.relevant_context` (`memory.py:38`) uses `db.search_memory` (FTS5 bm25, `db.py:3108`) + pinned only. RAG went hybrid; memory never did. A user preference phrased differently from the query won't surface. Skills retrieval (`runtime.py:_skill_context:2726` → `db.search_skills`) has the same lexical-only limitation. All the parts exist — the embedding client (`llm.py:266 embed`), RRF (`documents.py:268–276`), and normalized BLOB storage (`db.py:112`) — they're just not wired into memory/skills.

### Where (verified)
* `statera/memory.py:38 relevant_context`, `statera/db.py:3108 search_memory`, `:2742 save_chunk_embeddings` (reusable pattern), `runtime.py:2726 _skill_context`, `:1448` (memory injected into `_build_messages`).
* New tables: `memory_embeddings(memory_id, model, embedding BLOB)`, `skill_embeddings(...)` — mirror `document_chunks.embedding` storage.

### Diff sketch
```python
# memory.py — hybrid retrieval mirroring HybridRAG.retrieve's RRF.
async def relevant_context_hybrid(self, query, embedding_model, client, limit=8) -> str:
    lexical = self.db.search_memory(query, limit=limit, statuses=("approved","pinned"))
    semantic = []
    if embedding_model:
        qvec = (await client.embed([query], embedding_model))[0]
        semantic = self.db.search_memory_semantic(qvec, embedding_model, limit=limit)  # numpy dot, A1-style
    scores, rows = {}, {}
    for rank, m in enumerate(lexical):  rows[m["id"]] = m; scores[m["id"]] = scores.get(m["id"],0)+1/(60+rank)
    for rank, m in enumerate(semantic): rows[m["id"]] = m; scores[m["id"]] = scores.get(m["id"],0)+1/(60+rank)
    ranked = sorted(rows.values(), key=lambda m: scores[m["id"]], reverse=True)[:limit]
    return self._format_context(pinned=self.db.list_pinned_memory(6), relevant=ranked)
```
Memories get embedded lazily on approval (hook in `app.py:update_memory` → status `approved`) and skills on save (`db.save_skill`). Fallback to today's lexical `relevant_context` when no embedding model configured.

### Acceptance criteria
* A paraphrased query surfaces a semantically-matching approved memory that bm25 misses (fixture test).
* No embedding model configured → identical to today's `relevant_context`.
* Embedding backfill is idempotent and off the hot path (lazy on approval, not on retrieval).

### Test plan
`tests/test_hybrid_memory_v11.py`: paraphrase recall; lexical fallback; skills hybrid; backfill idempotency. Reuse `test_storage_memory_tools.py` fixtures.

---

## A12 — Pluggable search backend + readability extractor

**Merges:** S2-1.6. **Effort:** M · **Impact:** M · **Risk:** M. **Decision:** D5.

### What
Make the search backend pluggable: use a real search API (Brave/Tavily/SearXNG) when a key is configured; keep the DuckDuckGo→Bing HTML scraper as the no-key/offline fallback. Add a readability-style main-content extractor for `read_url` to strip nav/boilerplate.

### Why
`_search_results` (`tools.py:987`) regex-scrapes DuckDuckGo's HTML then falls back to Bing (`:1021`), parsing with regex — this breaks whenever either site tweaks markup. `read_url` page extraction uses a hand-written `_ReadableHTMLParser` (`tools.py:204`) that keeps boilerplate, inflating tokens and diluting relevance. A real API is robust; readability extraction makes page text cheaper and more accurate.

### Where (verified)
* `statera/tools.py:987 _search_results`, `:931 _web_search`, `:944 _research_web`, `:891 _read_url`, `:204 _ReadableHTMLParser`.
* Config: extend `CommandSettings` or a new `SearchSettings` in `schemas.py` for API provider + key (stored like other secrets).
* All egress must remain on the SSRF-pinned `self._http_client` (`tools.py:294`).

### Diff sketch
```python
# tools.py — strategy dispatch; API path still uses the pinned client.
def _search_results(self, query, limit):
    provider = self.db.get_settings().search  # new SearchSettings: provider, api_key
    if provider.api_key and provider.name in {"brave","tavily"}:
        try:    return self._search_api(provider, query, limit)   # pinned httpx GET to api.search.brave.com etc.
        except Exception as exc:
            _LOGGER.warning("Search API failed, falling back to scraper: %s", exc)
    return self._search_scrape(query, limit)                      # today's DDG→Bing regex path, unchanged
```
```python
# read_url readability: prefer <article>/<main>, drop nav/aside/footer/script, density heuristic.
def _extract_readable(self, html_text):
    doc = _MainContentExtractor()      # keeps largest text-dense block; falls back to _ReadableHTMLParser
    doc.feed(html_text)
    return doc.text() or _ReadableHTMLParser_text(html_text)
```
API endpoints must be added to the SSRF allow-reasoning (they resolve to public IPs, so the pin passes them; no loopback exception needed).

### Acceptance criteria
* No key configured → byte-identical scraper behavior (existing search tests pass).
* With a mocked API key, results come from the API; API failure transparently falls back to scraper.
* Readability extraction reduces token count on a boilerplate-heavy fixture ≥30 % while retaining the main article text.
* All search/fetch egress remains on the pinned client (SSRF test still blocks loopback).

### Test plan
Extend `tests/test_research_memory_uploads.py`: API-vs-scraper dispatch; fallback on API error; readability token-reduction on a fixture; SSRF pin unchanged.

---

## A13 — Hierarchical, token-aware auto-compaction

**Merges:** S2-1.7. **Effort:** M · **Impact:** M · **Risk:** M. **Depends on A3.**

### What
Replace the lossy char-truncation auto-compaction with tiered rolling summaries (recent verbatim → mid-term model summary → long-term facts) and token-accurate budgeting via A3's `TokenCounter`.

### Why
The manual path (`compact_session`, `runtime.py:1073`) already produces a proper model-generated checkpoint — but the **automatic** post-run path `_compact_session_context` (`runtime.py:1612`) just concatenates char-truncated lines (`_summarize_text(..., 180)` at `:1625`), a lossy heuristic, and its budgeting is char-based. Tiered summaries preserve far more signal at the same budget.

### Where (verified)
* `statera/runtime.py:1612 _compact_session_context` (auto path, called at `:838`), `:1073 compact_session` (manual, model-generated — reuse its machinery), `:1582/:1589` char budgets.

### Diff sketch
```python
# runtime.py — auto-compaction reuses the model checkpoint, tiered by token budget.
async def _compact_session_context_v2(self, session_id, counter, force=False):
    history = self.db.list_messages(session_id, limit=200)
    if counter.count_messages(history) < settings.model.context_budget * 0.6 and not force:
        return ""                                   # only compact when genuinely near budget
    recent = self._take_recent_under(history, counter, budget=settings.model.context_budget * 0.25)  # verbatim
    mid    = [m for m in history if m not in recent]
    summary = await self.client(json_settings).complete_json(  # model summary, not char-trunc
        [{"role":"system","content":COMPACTION_PROMPT}, {"role":"user","content":self._render(mid)}],
        max_tokens=settings.model.compaction_summary_tokens, schema=COMPACTION_SCHEMA)  # A10
    return self._merge_tiers(previous_longterm=session.context_summary, mid_summary=summary, recent=recent)
```
Long-term facts tier can feed A11/B8. Keep the char-trunc path as a no-model fallback when no model configured (today's behavior).

### Acceptance criteria
* Auto-compaction output is model-generated (not char-truncated) when a model is configured; retains key entities a char-trunc drops (fixture with a named constraint mid-thread).
* Token-budget-aware (uses A3 counter, respects `context_budget`).
* No model → falls back to today's `_compact_session_context`.
* Manual `compact_session` behavior unchanged.

### Test plan
`tests/test_compaction_v11.py`: signal-retention (a fact stated early survives compaction); token-budget adherence; no-model fallback.

---

## A14 — Windowed `read_file` + loop-nesting documentation

**Merges:** S2-1.8 (windowing + docs halves). **Effort:** S · **Impact:** M · **Risk:** L.

### What
Add `offset`/`limit` (byte or line range) to `read_file` so the agent can page large files instead of pulling whole files. Separately, document the three nested run loops so their budget interactions are legible.

### Why
`_read_file` (`tools.py:472`) only truncates from the start via `max_chars` — no offset/line-range, so reading line 4 000 of a 6 000-line file means pulling the whole prefix. Windowed reads cut tokens hard on large files. Separately, `run_session` nests three loops — outer `while unlimited_tool_use or turn < max_turns` (`runtime.py:582`), inner `while ... tool_turn < max_tool_turns` (`:1681`), and the judge-continue path (`:696`) — whose budgets (`tool_budget` vs `max_turns` vs `max_tool_turns`) interact subtly; a module docstring + a diagram prevents off-by-one surprises.

### Where (verified)
* `statera/tools.py:472 _read_file`, `:1528 TOOL_DEFINITIONS["read_file"]` (add params to schema), `runtime.py:1992` (missing-required validation for read_file).
* `runtime.py:582`, `:1681`, `:696` (the three loops) → add a `run_session` docstring.

### Diff sketch
```python
# tools.py _read_file — optional line window; default unchanged.
def _read_file(self, session_id, arguments, settings):
    path = self._resolve_workspace_path(session_id, arguments.get("path") or arguments.get("filename") or "", settings)
    if not path.is_file(): raise FileNotFoundError(str(path))
    max_chars = max(500, min(200_000, int(arguments.get("max_chars") or 20_000)))
    start_line = max(0, int(arguments.get("start_line") or 0))
    max_lines  = int(arguments.get("max_lines") or 0)
    text = path.read_bytes()[: max(max_chars * 4, 4096)].decode("utf-8", errors="replace")
    if start_line or max_lines:
        lines = text.splitlines()
        end = start_line + max_lines if max_lines else len(lines)
        text = "\n".join(lines[start_line:end])
        return f"[lines {start_line}-{min(end, len(lines))} of {path.name}]\n" + text[:max_chars]
    return text[:max_chars]
```
Extend the `read_file` schema (`tools.py:1533`) with `start_line`/`max_lines` integers.

### Acceptance criteria
* Reading `{start_line: N, max_lines: K}` returns exactly those lines with a header; omitting them reproduces today's behavior byte-for-byte.
* Windowed read of a large file uses far fewer tokens than a full read (assert).
* `run_session` gains a docstring enumerating the three loops and their budget guards.

### Test plan
`tests/test_read_window_v11.py`: line-range correctness; default-path parity; out-of-range clamping. Doc-lint: docstring present.

---

# Track B — Deep dives (radical)

---

## B1 — Tiered model routing for plumbing calls

**Merges:** S2-2.2. **Effort:** M · **Impact:** H · **Risk:** M. **Depends on A2 (cache), A10 (schemas). Partially built already.**

### What
Route the "bookkeeping" model calls — judge, turn classifier (A9), memory extraction, compaction, reflection — to a cheap/fast model (or the native llama.cpp slot), reserving the strong model for the actual act loop.

### Why
One configured model does everything today. The judge (`runtime.py:2300`) can fire after every turn; reflection is ~1 600 tokens (`:1014`); memory extraction, compaction, and conversational replies all use the primary model. **Crucially, the infrastructure already exists:** `run_session` already selects `json_settings` for the judge client (`runtime.py:416,419`) and `coder_settings` for the act loop (`:415,626`), and `select_model_settings` (`model_routing.py:25`) resolves `primary/planner/coder/json/embedding` profiles — but routing is gated off by default (`experimental.model_routing_enabled`) and reflection/memory/compaction still use the primary `model_client` (`:789`) rather than the `json` route. This item finishes the wiring and adds a documented "cheap route" recommendation.

### Where (verified)
* `statera/model_routing.py:25 select_model_settings`, `:18 route_profile`; `schemas.py:70 ModelRoutingSettings`, `:85 ExperimentalSettings`.
* `runtime.py:413–420` route selection; `:789 memory.consolidate(model_client=...)` (should be json route); `:1002/:1083` reflection & compaction already use `json_settings` ✓; `:2300 _judge` already uses `json` client ✓.

### Diff sketch
```python
# runtime.py run_session — send memory extraction through the json/cheap route too.
memory_settings, memory_route = select_model_settings(settings, "json")
memory_client = self.client(memory_settings)
...
created = await self.memory.consolidate(session_id, memory_client, 12, 2, final_history)  # was model_client
```
```python
# model_routing.py — add an explicit "utility" fallback chain: json -> planner -> primary.
def select_utility_settings(app_settings, prefer=("json","planner")):
    for route in prefer:
        settings, diag = select_model_settings(app_settings, route)
        if diag["active"]:
            return settings, diag
    return app_settings.model, {"route": "utility", "active": False, "fallback": True}
```
Document (in Settings help + README) the recommended pattern: strong model on `primary`/`coder`, a small fast model (or native slot) on `json`. No new provider is required (**D4**).

### Acceptance criteria
* With routing enabled, judge/classify/memory/compaction/reflection all hit the `json` route; act loop hits `coder`/`primary` (assert via diagnostics `routing`).
* With routing disabled, every call uses the primary model exactly as today.
* A profile with an empty model falls back to primary (existing `select_model_settings` diagnostics: `profile_incomplete`).

### Test plan
Extend `tests/test_model_urls.py`/`test_v10_capabilities.py`: route-dispatch matrix (enabled/disabled), fallback-on-incomplete-profile, diagnostics correctness.

---

## B2 — Real multi-agent orchestration

**Merges:** S1-2.1, S2-2.3. **Effort:** XL · **Impact:** VH · **Risk:** H. **Depends on B1, B4, A9. Scaffold already present.**

### What
A lightweight `Orchestrator` that (1) asks the `planner` route for a DAG of sub-tasks with dependencies, (2) schedules independent nodes concurrently as child `SessionRuntime` runs (reusing `start_task_run`), (3) uses the blackboard as the shared memory bus, (4) runs a verifier pass against the original objective. Gate sub-agent write access so only the verifier/primary commits mutations. Start read-only (research/analysis fan-out), graduate to code tasks.

### Why
The biggest capability multiplier. The primitives already exist: run steps, the blackboard (`runtime.py:2721 _add_blackboard`), project plans (`project_plans.py`), model routing with distinct planner/coder/json profiles (`model_routing.py`), per-session locking (`runtime.py:124 _session_lock`), and an event bus. Today "orchestration" is a **stub**: `/api/experimental/orchestrator/runs` (`app.py:1264`) creates a *blocked* run with five role steps and no execution, and the "subagent" is literally a second webview window (`main.py:140 open_subagent`) with no delegation. Decomposing into independent sub-goals that run concurrently turns Ilo from one loop into a team, and each sub-agent carries a small focused context instead of one giant thread (efficiency win too).

### Where (verified)
* `statera/app.py:1264 start_experimental_orchestrator` (the scaffold to replace), `schemas.py:88 multi_agent_enabled`.
* `statera/runtime.py:204 start_task_run`, `:410 run_session`, `:124 _session_lock`, `:2721 _add_blackboard`, `:458–461` orchestration diagnostics.
* `statera/project_plans.py` (DAG rendering already exists), `model_routing.py:25` (planner route).
* `main.py:140 open_subagent` (UI-only today).

### Diff sketch
```python
# statera/orchestrator.py (new) — bounded planner → parallel workers → verifier.
class Orchestrator:
    def __init__(self, runtime): self.rt = runtime
    async def run(self, session_id, objective, agent_id):
        plan = await self._plan_dag(objective)                       # planner route -> {nodes, edges}
        results: dict[str, dict] = {}
        for layer in self._topo_layers(plan):                        # independent nodes per layer
            layer_runs = [self._spawn_child(session_id, node, read_only=True) for node in layer]
            for node, res in zip(layer, await asyncio.gather(*layer_runs)):
                await self.rt._add_blackboard(session_id, res["run_id"], f"subtask:{node['id']}", res["summary"])
                results[node["id"]] = res
        verdict = await self._verify(objective, results)             # verifier route, reconciles blackboard
        if verdict["approved"]:
            await self._commit(session_id, results)                  # only here are mutations allowed
        return verdict

    async def _spawn_child(self, session_id, node, read_only):
        run = await asyncio.to_thread(self.rt.db.create_task_run, {..., "tool_budget": node["budget"],
            "agent_id": node["agent_id"], "diagnostics": {"orchestration": {"role": node["role"],
            "write_gated": read_only}}})
        # child run uses _effective_tool_names filtered to read-only when write_gated
        return await self.rt.run_session_child(session_id, node["agent_id"], run["id"], write_gated=read_only)
```
Add `write_gated` plumb-through to `_effective_tool_names` (`runtime.py:2455`) so gated children get the read-only tool subset. Reuse the existing `orchestration` diagnostics block (`:458`). Replace the `app.py:1264` stub with a call into `Orchestrator.run`, still behind `multi_agent_enabled`.

### Acceptance criteria
* A research fan-out ("compare X, Y, Z") spawns ≥2 concurrent read-only children, each writing findings to the blackboard, with a verifier reconciliation — and mutates **zero** project files.
* Write-gated children cannot call mutating tools (tool list filtered; attempt → blocked_result).
* DAG scheduler runs independent nodes concurrently and dependent nodes in order (topo test).
* Feature stays fully behind `experimental.multi_agent_enabled`; disabled path is today's single-agent runtime.
* **Security gate:** no path lets a sub-agent escalate write access; SSRF/path guards unchanged for all children.

### Test plan
`tests/test_orchestrator_v11.py`: DAG topo-order; concurrency; write-gating enforcement; verifier reject → no commit; disabled-flag parity. Fake models for planner/worker/verifier.

---

## B3 — ANN vector store behind `HybridRAG`

**Merges:** S1-2.2, S2-2.5 (ANN half). **Effort:** L · **Impact:** H · **Risk:** M. **Depends on A1. Decision:** D2.

### What
Adopt `sqlite-vec` (stays single-file, no new service) for an ANN index behind the existing `HybridRAG.retrieve` signature. Scales retrieval from "a few docs" to a real corpus and unblocks cross-session semantic recall + B8.

### Why
Even with A1's NumPy matrix, scoring is still O(N·d) — fine to ~10k chunks, but there's no ANN index and no cross-session semantic recall. `sqlite-vec` gives sub-linear ANN while preserving the zero-server, single-file SQLite ethos and the PyInstaller packaging story (D2). Wrapped behind `retrieve()`, nothing upstream changes.

### Where (verified)
* `statera/documents.py:239 retrieve` (the wrap point), `:264 document_chunks` (replaced by an ANN query), `db.py:2742 save_chunk_embeddings` (also upsert into the vec index), `:2483 document_chunks`.
* New: a `vec_document_chunks` virtual table via the `sqlite-vec` extension loaded in `Database.__init__` (`db.py:148`).

### Diff sketch
```python
# db.py __init__ — load the extension (feature-detected).
try:
    self._conn.enable_load_extension(True); import sqlite_vec; sqlite_vec.load(self._conn)
    self._conn.execute("CREATE VIRTUAL TABLE IF NOT EXISTS vec_chunks USING vec0(embedding float[768])")
    self._have_vec = True
except Exception:
    self._have_vec = False          # degrade to A1's numpy matrix path
```
```python
# db.py — ANN query; A1 numpy path remains the fallback.
def search_chunks_ann(self, session_id, query_vec, model, limit):
    if not self._have_vec: return None
    return self._read_fetchall(
        "SELECT c.*, d.name FROM vec_chunks v JOIN document_chunks c ON c.rowid=v.rowid "
        "JOIN documents d ON d.id=c.document_id WHERE c.session_id=? AND c.embedding_model=? "
        "AND v.embedding MATCH ? ORDER BY distance LIMIT ?", (session_id, model, _pack(query_vec), limit))
```
`retrieve()` prefers `search_chunks_ann`, else A1 matrix, else pure-Python — a three-tier graceful chain. `save_chunk_embeddings` upserts into `vec_chunks` alongside the BLOB.

### Acceptance criteria
* With `sqlite-vec` present, top-k recall vs. exact numpy is ≥0.95 on a seeded corpus.
* Without the extension, falls back to A1 numpy with identical results to today.
* Index stays consistent across add/delete/re-embed (rebuild-on-drift test).
* Single-file DB still opens/backs up cleanly (migration + backup tests pass).

### Test plan
`tests/test_ann_store_v11.py`: recall@k vs exact; fallback parity; consistency on mutation; extension-absent path.

---

## B4 — Code-aware workspace intelligence

**Merges:** S2-2.5 (code half). **Effort:** L · **Impact:** H · **Risk:** M. **Extends existing `project_index.py`. Decision:** D3. **Feeds B2, B7.**

### What
Add tree-sitter symbol extraction + code-aware chunking so the agent navigates large repos by symbol, not regex. Upgrade the existing `find_symbol`/`analyze_impact`/`workspace_map` tools from heuristic string matching to real AST symbols and import graphs.

### Why
`search_files` (`tools.py:573`) is regex-over-files; the existing code index (`project_index.py`, `ProjectIndexer.scan`) and its tools `workspace_map` (`tools.py:684`), `find_symbol` (`:726`), `analyze_impact` (`:741`) already exist but symbol/import extraction is heuristic (substring matching on `row["symbols"]`/`row["imports"]`). Real tree-sitter parsing turns Ilo from a doc-Q&A tool into an agent that can navigate big repos by definition/reference — the substrate for orchestrated code tasks (B2) and the verify loop (B7).

### Where (verified)
* `statera/project_index.py` (194 lines, `ProjectIndexer`), `statera/tools.py:684 _workspace_map`, `:726 _find_symbol`, `:741 _analyze_impact`, `db.py:save_project_index`/`find_project_symbols`/`list_project_index_files`.

### Diff sketch
```python
# project_index.py — pluggable extractor; tree-sitter for top langs, regex fallback (today's path).
class SymbolExtractor:
    def __init__(self):
        try:
            import tree_sitter_languages as tsl
            self._parsers = {lang: tsl.get_parser(lang) for lang in ("python","javascript","typescript","go","rust")}
        except Exception:
            self._parsers = {}                 # degrade to regex heuristic (current behavior)
    def extract(self, path, source, language):
        parser = self._parsers.get(language)
        if not parser: return self._regex_symbols(source, language)   # today's logic
        tree = parser.parse(source.encode("utf-8"))
        return self._walk(tree, language)      # defs, refs, imports with byte ranges
```
`ProjectIndexer.scan` calls the extractor; `_find_symbol`/`_analyze_impact` consume precise symbol/import edges. Byte ranges enable B7's node-range edits.

### Acceptance criteria
* `find_symbol` returns exact definition locations (file+range) for the 5 supported languages on a fixture repo.
* `analyze_impact` uses real import edges (dependents computed from parsed imports, not substring).
* No tree-sitter installed → regex fallback reproduces today's index behavior.
* Index scan stays bounded (`max_files`/`max_file_bytes` respected).

### Test plan
`tests/test_code_index_v11.py`: symbol accuracy per language; import-graph dependents; regex fallback parity; scan bounds.

---

## B5 — MCP client support

**Merges:** S2-2.6. **Effort:** L · **Impact:** H · **Risk:** H. **Decision:** D6. **Security-sensitive.**

### What
Add an MCP client so Ilo can consume any external MCP server (GitHub, DBs, browsers, cloud APIs) and expose its tools through the existing OpenAI-compatible tool-calling loop — no bespoke integration per tool.

### Why
Ilo hand-codes each tool in `TOOL_DEFINITIONS` (`tools.py:1516`, 23 tools). An MCP client is the highest capability-per-line addition available and future-proofs the tool layer. Given the streaming tool-calling loop already in place (`runtime.py:1732`), MCP tools slot in as additional definitions + a dispatch branch.

### Where (verified)
* `statera/tools.py:1516 TOOL_DEFINITIONS`, `:317 definitions`, `:341 execute` (the if/elif dispatch to extend), `tool_names.py:3 TOOL_NAMES` (registry), `runtime.py:1657 self.tools.definitions(tool_names)`.
* Config: new `McpServerSettings` in `schemas.py` (command/args/transport/enabled), gated by a new `experimental.mcp_enabled`.

### Diff sketch
```python
# statera/mcp_client.py (new) — manage stdio MCP servers, adapt tools to OpenAI schema.
class McpManager:
    async def start(self, servers: list[McpServerSettings]): ...      # spawn stdio, handshake, list_tools
    def openai_tool_defs(self) -> dict[str, dict]:                     # {f"mcp__{srv}__{tool}": {...schema}}
        return {self._qualified(s, t): self._to_openai_schema(t) for s, tools in self._tools.items() for t in tools}
    async def call(self, qualified_name, args) -> ToolResult: ...      # route to the owning server

# tools.py execute — dispatch mcp__* to the manager (still inside the same try/redaction/clip envelope).
elif tool_name.startswith("mcp__"):
    result = self._loop.run_until_complete(self._mcp.call(tool_name, arguments))
```
`ToolRegistry.definitions` merges `TOOL_DEFINITIONS | mcp.openai_tool_defs()`. **Security:** MCP servers are external processes — each spawned server runs under the same env allow-list discipline as `_command_environment` (`tools.py:1493`); any HTTP/SSE transport (D6) routes through SSRF review; MCP tool output passes through the same redaction + `_clip_output` (`tools.py:408`) envelope. Off by default; user explicitly enables each server.

### Acceptance criteria
* A stdio MCP server's tools appear as `mcp__<server>__<tool>` and are callable through the normal loop.
* MCP disabled by default; no server auto-starts.
* MCP tool output is redacted + clipped like native tools; failures surface as `blocked_result`, never crash the loop.
* **Security gate:** MCP processes inherit only the env allow-list; remote transports pass SSRF pinning; a malicious tool schema can't inject beyond its namespace.

### Test plan
`tests/test_mcp_client_v11.py`: fake stdio server handshake + tool listing; qualified-name dispatch; redaction/clip; disabled-by-default; env-allowlist for spawned server.

---

## B6 — Measured self-improvement eval harness

**Merges:** S1-2.3, S2-2.8. **Effort:** L–XL · **Impact:** H · **Risk:** M. **Depends on A10. Land before B2/B7.**

### What
Build a versioned eval set (objectives + rubric/graded checks). When a prompt/tool/skill/improvement change is proposed, run the eval set A/B (control vs. candidate) using the existing judge as grader, and only surface candidates that measurably win. Regression-protects the agent itself.

### Why
Reflection produces memory/skill/improvement candidates for human review (`runtime.py:1266 _save_reflection_candidates`) and `_apply_improvement` (`app.py:1619`) persists approved ones — but **nothing measures whether an accepted improvement actually helps.** The `/api/eval-lab/run` endpoint (`app.py:1241`) returns a **hardcoded "passed"** with two fake scenarios. "Self-improvement" today is unquantified suggestion-making. A scored A/B loop makes it real and gives regression protection on the agent — the thing most agent projects lack, and the safety net B2/B7 need.

### Where (verified)
* `statera/app.py:1241 run_eval_lab` (the fake to replace), `:1619 _apply_improvement`, `:1565 update_improvement` (approval hook), `schemas.py:429 EvalRunIn`, `:90 eval_lab_enabled`.
* Grader: reuse `_judge` (`runtime.py:2300`) / structured grading (A10).

### Diff sketch
```python
# statera/eval_lab.py (new) — real A/B eval with graded rubric.
class EvalHarness:
    async def run(self, name, control_cfg, candidate_cfg) -> dict:
        cases = self._load_eval_set(name)                      # versioned JSON: [{objective, rubric, checks}]
        control  = await self._run_suite(cases, control_cfg)   # baseline agent config
        candidate = await self._run_suite(cases, candidate_cfg)
        return {"name": name, "control": control, "candidate": candidate,
                "delta": candidate["score"] - control["score"],
                "regressions": self._regressions(control, candidate),
                "recommend": candidate["score"] > control["score"] and not self._regressions(control, candidate)}

    async def _grade(self, case, transcript) -> float:         # judge-as-grader against rubric + hard checks
        hard = all(chk(transcript) for chk in case["checks"])  # e.g. "SSRF still blocks loopback"
        return 0.0 if not hard else await self._rubric_score(case["rubric"], transcript)
```
Wire `update_improvement` (`app.py:1565`) so approving an improvement can optionally trigger an A/B run and record the delta on the improvement payload. Ship a starter eval set that **includes security-invariant checks** (SSRF, path escape, redaction) so §5's "no security regression" becomes machine-enforced.

### Acceptance criteria
* Eval set runs control vs. candidate and reports a scored delta + regression list (not a hardcoded pass).
* A deliberately-worse candidate prompt scores lower and is **not** recommended.
* Security-invariant checks are hard gates: any candidate that weakens SSRF/path/redaction fails regardless of task score.
* Off by default (`eval_lab_enabled`); mutates no project files.

### Test plan
`tests/test_eval_harness_v11.py`: better/worse candidate discrimination; hard security-check veto; determinism with fake models; no-file-mutation.

---

## B7 — AST edits + auto-verify edit→test→fix loop

**Merges:** S1-2.4, S2-2.9. **Effort:** L–XL · **Impact:** H · **Risk:** M. **Depends on B4. Feeds coding tasks.**

### What
(a) Add structured edits (tree-sitter node-range replacement) alongside the exact-string path; (b) add a real unified-diff apply tool; (c) when `commands.enabled`, add an opt-in post-edit verification step that runs the project's test/lint command and feeds failures back into the loop before reporting done — a tight edit→test→fix cycle.

### Why
Editing is exact-string only: `replace_in_file` (`tools.py:643`), `propose_patch`/`apply_patch_file` (`tools.py:821/849`) all require exact `old_text` match and `write_file` (`:614`) rewrites whole files — fragile on large/ambiguous files. Verification is opportunistic (visual capture, optional commands) with no automatic "did my edit break the build/tests?" gate. The diagnostics already track `files_touched`/`verification_commands` (`runtime.py:1055–1069`), so the plumbing to feed results back exists. This is what separates "generates code" from "lands working code."

### Where (verified)
* `statera/tools.py:643 _replace_in_file`, `:821 _propose_patch`, `:849 _apply_patch_file`, `:614 _write_file`.
* `runtime.py:1055 _merge_run_stats` (already tracks `files_touched`/`verification_commands`/`patch_preconditions_failed`), `:2108–2129` (per-tool verification capture).
* B4's byte-range symbols enable node-range edits.

### Diff sketch
```python
# tools.py — real unified-diff apply (git-style), not just string replace.
def _apply_unified_diff(self, session_id, arguments, settings):
    self._require_workspace_writes()
    path = self._resolve_workspace_path(session_id, arguments["path"], settings)
    patched = _apply_patch(path.read_text("utf-8", "replace"), arguments["diff"])  # hunk matcher w/ fuzz
    self._revalidate_workspace_path(session_id, path, settings).write_text(patched, "utf-8")
    return f"Applied {arguments['diff'].count('@@')//2} hunk(s) to {self._relative(session_id, path, settings)}"
```
```python
# runtime.py — post-edit verify loop (opt-in, commands.enabled).
if files_touched and app_settings.commands.verify_after_edit and app_settings.commands.enabled:
    verify_cmd = app_settings.commands.verify_command            # e.g. "pytest -q" / "npm test"
    res = await asyncio.to_thread(self.tools.execute, session_id, "run_command", {"command": verify_cmd}, ...)
    if res.preview.startswith("Tool failed") or _has_failures(res.preview):
        messages.append({"role": "user", "content": VERIFY_FAILED_PROMPT + res.preview[:4000]})
        continue                                                 # feed failures back before reporting done
```
Structured edits: a `replace_symbol` tool using B4 byte-ranges (`{path, symbol, new_body}`) as an alternative to fragile string matching.

### Acceptance criteria
* Unified-diff apply lands multi-hunk patches; malformed/ambiguous hunks fail cleanly (no partial write).
* `replace_symbol` edits by AST node range for supported languages; falls back to string path otherwise.
* With `verify_after_edit`, a failing test injected mid-task triggers a fix cycle and the run doesn't report "done" until green (or budget exhausted).
* Verify loop is off by default and requires `commands.enabled`.
* **Security gate:** all edit paths still route through `_resolve_workspace_path` + `_revalidate_workspace_path`; verify command runs under the env allow-list.

### Test plan
`tests/test_verify_edit_v11.py`: multi-hunk apply; ambiguous-hunk rejection; symbol-range edit; verify-fail→fix→pass cycle; off-by-default; path-guard intact.

---

## B8 — Structured memory graph

**Merges:** S1-2.5. **Effort:** L · **Impact:** M–H · **Risk:** M. **Depends on B3. Pairs with A11.**

### What
Layer a lightweight entity/relation store (entities, attributes, edges) populated during consolidation and retrieved by semantic + graph expansion. Keep the flat store as fallback so nothing regresses.

### Why
Long-term memory is flat text records surfaced by keyword relevance (`memory.py:38 relevant_context`) with string-similarity dedup (`_matches_duplicate:197`, `SequenceMatcher > 0.9`). Flat memory can't answer "what do I know about entity X and how does it relate to Y," and dedup is string-only. A graph layer improves recall quality and dedup precision.

### Where (verified)
* `statera/memory.py:70 consolidate` (extraction point), `:38 relevant_context` (retrieval), `:197 _matches_duplicate`, `db.py:add_memory`/`search_memory:3108`.
* New tables: `mem_entities(id, name, type, embedding)`, `mem_edges(src, dst, relation, weight, source_memory_id)`.

### Diff sketch
```python
# memory.py consolidate — also extract entities/relations from accepted facts.
for candidate in valid_candidates:
    saved = await asyncio.to_thread(self.db.add_memory, {...})              # unchanged flat record
    triples = await self._extract_triples(candidate["text"], json_client)  # (entity, relation, entity)
    await asyncio.to_thread(self.db.upsert_graph, saved["id"], triples)     # entities + edges
```
```python
# memory.py relevant_context_graph — semantic seed + 1-hop expansion, fused with flat retrieval.
seeds = self.db.search_entities_semantic(query_vec, limit=5)               # B3 ANN over entity embeddings
expanded = self.db.expand_edges([e["id"] for e in seeds], hops=1)
facts = self.db.memories_for_entities([n["id"] for n in seeds + expanded])
return self._fuse(flat=self.relevant_context_hybrid(...), graph=facts)     # RRF, flat as fallback
```

### Acceptance criteria
* A relational query ("what relates to project X") returns graph-expanded facts a flat keyword search misses.
* Entity dedup merges "Ilo" / "the Ilo project" better than string-similarity alone (precision test).
* No graph data / no ANN → falls back to A11 hybrid (or today's flat) retrieval with no error.

### Test plan
`tests/test_memory_graph_v11.py`: triple extraction; 1-hop expansion recall; entity-merge precision; flat fallback.

---

## B9 — Speculative decoding + KV-cache reuse (native runtime)

**Merges:** S1-2.6. **Effort:** M · **Impact:** H (local users) · **Risk:** M. **Settings already exist.**

### What
Pass `--model-draft`/speculative flags through to `llama-server` when a draft model is configured, and keep the server's prompt cache warm by preserving prefix stability between turns (synergizes with A2).

### Why
`NativeRuntimeSettings` **already carries** `speculative_decoding_enabled`, `draft_model_path`, `cache_prompt`, and `parallel_slots` (`schemas.py:131–133`), and the status endpoint reports `draft_model_ready` (`native_runtime.py:126–131`) — but `_server_args` (`native_runtime.py:377`) only wires `--ctx-size/--n-gpu-layers/--threads/--batch-size/--ubatch-size/--parallel`; it **never passes** the draft-model or prompt-cache flags. Speculative decoding (draft + verify) can 1.5–3× local throughput; KV-cache reuse cuts prefill when A2's stable prefix is in place. Expose as an advanced toggle (consistent with the off-by-default philosophy).

### Where (verified)
* `statera/native_runtime.py:377–415 _server_args` (add flags), `:124–131` status (already surfaces draft readiness), `schemas.py:114 NativeRuntimeSettings`.

### Diff sketch
```python
# native_runtime.py _server_args — append draft-model + prompt-cache flags when configured.
if settings.speculative_decoding_enabled and settings.draft_model_path:
    draft = Path(settings.draft_model_path).expanduser().resolve()
    if draft.is_file() and draft.suffix.lower() == ".gguf":
        args.extend(["--model-draft", str(draft)])
        if settings.draft_n_predict: args.extend(["--draft", str(settings.draft_n_predict)])   # new field
if settings.cache_prompt:
    args.append("--cache-reuse"); args.extend(["--cache-reuse", "256"])   # keep prefix KV warm (confirm flag name vs llama-server version)
```
> Confirm exact llama-server flag names against the bundled binary version at implementation time (they've churned: `--model-draft`, `--draft-max/--draft-min`, `--cache-reuse`). Add a `draft_n_predict` field to `NativeRuntimeSettings` if exposing tuning.

### Acceptance criteria
* With a draft model configured + toggle on, `_server_args` includes the speculative flags; off → today's args exactly.
* Invalid/missing draft path is ignored (no crash; status shows `draft_model_ready: false`).
* Prompt-cache flag only added when `cache_prompt` true.
* A local throughput benchmark shows improvement with a real draft model (documented, not asserted in CI).

### Test plan
`tests/test_native_runtime_v11.py`: arg-building matrix (draft on/off, valid/invalid path, cache on/off); no-regression on the existing arg set.

---

## B10 — Sandboxed command execution

**Merges:** S1-2.7, S2-2.10. **Effort:** XL · **Impact:** H · **Risk:** H. **Decision:** D7. **Security-defining.**

### What
Run `run_command`/`launch_alias` inside a constrained context — Windows Job Objects with UI/limit restrictions + a restricted token (or an opt-in WSL/container executor with a bind-mounted workspace and no host network) — so command tools can be enabled with far less risk, raising safe autonomy.

### Why
The README is candid: Ilo "does not attempt to sandbox software that is already running under the same Windows account" (`README.md:129`). Command tools are correctly off by default (`schemas.py:163 CommandSettings.enabled=False`), but that ceiling limits autonomy — the biggest blocker to hands-off runs. The env is already scrubbed to an allow-list (`tools.py:1493`) and processes are tracked/reaped (`:1227 _track_launched_process`, `:1246 _reap`, `:1250 _terminate`), so a real sandbox is an **extension of an existing boundary, not a rewrite.**

### Where (verified)
* `statera/tools.py:1079 _run_command` (Popen at `:1099`, `creationflags` at `:1090/1096`), `:1167 _launch_alias`, `:1227/:1246/:1250` process lifecycle, `:1493 _command_environment`.
* `main.py` / packaging for any WSL/container opt-in.

### Diff sketch
```python
# tools.py — wrap Popen in a Windows Job Object with restrictions (D7 default).
def _spawn_sandboxed(self, shell_args, cwd, env, creationflags):
    proc = subprocess.Popen(shell_args, cwd=cwd, env=env, shell=False,
                            stdout=PIPE, stderr=PIPE, creationflags=creationflags | CREATE_SUSPENDED)
    job = _create_restricted_job(                 # win32job: kill-on-close, no UI, memory/CPU caps, no breakaway
        active_process_limit=1, process_memory_limit=..., ui_restrictions=UILIMIT_ALL)
    _assign_process_to_job(job, proc)
    _resume_thread(proc)                          # run only after confinement is in place
    return proc, job
```
```python
# CommandSettings — new sandbox knobs (off by default; enabling command tools defaults sandbox ON).
class CommandSettings(BaseModel):
    ...
    sandbox_mode: Literal["off","job_object","container"] = "job_object"
    sandbox_no_network: bool = True
```
Container path (opt-in): execute inside a bind-mounted workspace with `--network none`. Either way, the env allow-list and workspace-scoped `cwd` (`tools.py:1086`) remain.

### Acceptance criteria
* With `sandbox_mode=job_object`, a spawned command that tries to outlive the job is killed on job close; UI/network restrictions verifiable.
* Sandbox failure to initialize → command is **refused**, not run unconfined (fail-closed).
* Existing env allow-list + workspace `cwd` scoping unchanged.
* Off/`container` paths documented; default flips to sandboxed when command tools are enabled.
* **Security gate:** a test proves a sandboxed command cannot reach the network (when `no_network`) or escape the workspace cwd.

### Test plan
`tests/test_sandbox_v11.py` (Windows-gated): job-object kill-on-close; fail-closed on sandbox init error; no-network enforcement; env-allowlist intact.

---

## B11 — Frontend decomposition + transcript virtualization

**Merges:** S1-2.8. **Effort:** L–XL · **Impact:** M now, H as a multiplier · **Risk:** L. **Decision:** D8.

### What
Extract Command Center, Blackboard, Live Activity, Composer, and the transcript from the monolith into components with a small state store (Zustand) over the existing SSE event stream; virtualize the message list (react-window) so 1 000-item threads stay smooth. Purely structural — no product behavior change.

### Why
`frontend/src/main.tsx` is **3 686 lines** and `styles.css` is **3 881 lines** (`wc -l` confirmed). It's the biggest maintainability risk and will bottleneck every future UI capability (the UI work implied by B2's orchestration view, B6's eval dashboard, B1's routing panel). Long threads also jank without list virtualization.

### Where (verified)
* `frontend/src/main.tsx` (3 686 lines), `frontend/src/styles.css` (3 881 lines), `frontend/src/components/` (exists — extraction target), `frontend/src/guide.tsx` (already split — the pattern to follow).
* SSE reducer: the event stream from `runtime.py` events (`item_*`, `run_*`, `tool_call_*`, `blackboard_updated`, `judge_*`).

### Diff sketch
```
frontend/src/
  store/session.ts            # Zustand: events -> normalized state (reduces the current in-component reducer)
  components/CommandCenter.tsx # extracted from main.tsx
  components/Blackboard.tsx
  components/LiveActivity.tsx
  components/Composer.tsx
  components/Transcript.tsx    # <FixedSizeList/VariableSizeList> from react-window
  main.tsx                     # ~300 lines: layout + providers only
```
Migrate incrementally: introduce the store, move one panel at a time behind identical DOM/CSS so screenshots match; virtualize the transcript last.

### Acceptance criteria
* No visual/behavioral change (screenshot diff of key states within tolerance; existing `test_ui_v81_contract.py` contract holds).
* `main.tsx` drops below ~500 lines; each panel is an isolated component with typed props.
* A 1 000-message thread scrolls at 60 fps (virtualization renders only visible rows).
* SSE event handling identical (same events → same state).

### Test plan
Extend `tests/test_ui_v81_contract.py` for the state-store contract; add a virtualization render test; manual screenshot parity on Command Center / Blackboard / Activity across the `orb`/`expanded` layout modes (`schemas.py:174`).

---

# 9. Appendices

## 9.1 File-by-file change index

| File | Items touching it | Nature of change |
|------|-------------------|------------------|
| `statera/documents.py` | A1, B3 | Cached numpy matrix + query LRU; ANN wrap behind `retrieve` |
| `statera/db.py` | A1, A7, A11, B3, B8 | `documents_signature`; table-driven migrations; memory/skill embeddings; `sqlite-vec` vtable; graph tables |
| `statera/runtime.py` | A2, A3, A4, A8, A9, A13, A14, B1, B2, B7 | Prompt reorder; token budgets; parallel tools; end-of-run concurrency; intent router; hierarchical compaction; loop docs; utility routing; orchestrator hooks; verify loop |
| `statera/llm.py` | A2, A5, A10 | `cache_control`/block system; retry/backoff; structured outputs |
| `statera/tools.py` | A6, A12, A14, B4, B5, B7, B10 | Result cache; pluggable search + readability; windowed read; AST index consumers; MCP dispatch; diff apply; sandbox |
| `statera/memory.py` | A11, B8 | Hybrid retrieval; graph extraction/expansion |
| `statera/model_routing.py` | B1 | Utility route chain |
| `statera/native_runtime.py` | B9 | Speculative/draft + cache flags in `_server_args` |
| `statera/project_index.py` | B4 | Tree-sitter symbol/import extraction |
| `statera/schemas.py` | A5, A12, B5, B9, B10 | `max_request_retries`; `SearchSettings`; `McpServerSettings`+`mcp_enabled`; `draft_n_predict`; sandbox knobs |
| `statera/app.py` | B2, B6 | Real orchestrator endpoint; real eval harness endpoint + approval A/B hook |
| `statera/context.py` | A3 | Token-aware tracking |
| `statera/tokens.py` (new) | A3 | `TokenCounter` |
| `statera/orchestrator.py` (new) | B2 | Planner→workers→verifier |
| `statera/eval_lab.py` (new) | B6 | A/B eval harness |
| `statera/mcp_client.py` (new) | B5 | MCP manager |
| `statera/schemas_json.py` (new) | A10 | JSON Schemas for judge/reflection/memory/compaction/router |
| `frontend/src/*` | B11 | Decomposition + virtualization |

## 9.2 Traceability matrix (every source sub-item → roadmap item)

| Source | → | Item | Source | → | Item |
|---|---|---|---|---|---|
| S1-1.1 | → | A4 | S2-1.1 | → | A1 |
| S1-1.2 | → | A1 | S2-1.2 | → | A11 |
| S1-1.3 | → | A2 | S2-1.3 | → | A5 |
| S1-1.4 | → | A3 | S2-1.4 | → | A4 |
| S1-1.5 | → | A8 | S2-1.5 | → | A9 |
| S1-1.6 | → | A9 | S2-1.6 | → | A12 |
| S1-1.7 | → | A6 | S2-1.7 | → | A3 + A13 |
| S1-1.8 | → | A7 | S2-1.8 | → | A14 + A6 |
| S1-2.1 | → | B2 | S2-2.1 | → | A2 |
| S1-2.2 | → | B3 | S2-2.2 | → | B1 |
| S1-2.3 | → | B6 | S2-2.3 | → | B2 |
| S1-2.4 | → | B7 | S2-2.4 | → | A9 |
| S1-2.5 | → | B8 | S2-2.5 | → | B3 + B4 |
| S1-2.6 | → | B9 | S2-2.6 | → | B5 |
| S1-2.7 | → | B10 | S2-2.7 | → | A10 |
| S1-2.8 | → | B11 | S2-2.8 | → | B6 |
| | | | S2-2.9 | → | B7 |
| | | | S2-2.10 | → | B10 |
| S2 "direction decisions" | → | Decision Register (§6) | | | |

## 9.3 Reproducing the empirical results

Both scripts live in the session scratchpad and are self-contained (stdlib + numpy):

* `profile_rag.py` — reproduces `_dot`/`_top_semantic` and the `db._encode/_decode_embedding` BLOB format (`struct <II` header + `<Nf` float32), then benchmarks pure-Python vs. cached-numpy across {500,2000,5000,10000}×{768,1024}. Output → §2.1.
* `profile_prompt.py` — verbatim copies of the static system-prompt blocks from `_build_messages` (`runtime.py:1487–1553`), measured in chars and two token estimators. Output → §2.2.

Run with the project venv: `./.venv/Scripts/python.exe <script>`. NumPy 2.5.0 is already present (transitive dep of onnxruntime/soundfile); no install needed for A1's feasibility.

## 9.4 What is *already built* (so we activate, not rebuild)

A recurring theme worth stating plainly for estimation: much of Track B is **wiring, not green-field.**

* **Model routing (B1):** `select_model_settings` + six route profiles exist; judge already uses `json`, act loop already uses `coder`. Just extend to memory/router and document the cheap-model pattern.
* **Orchestration (B2):** blackboard, run steps, project-plan DAG rendering, per-session locks, and an (empty) orchestrator endpoint + diagnostics block all exist. The subagent is UI-only today.
* **Code index (B4):** `project_index.py` + `workspace_map`/`find_symbol`/`analyze_impact` tools exist with heuristic extraction; B4 upgrades the extractor to tree-sitter.
* **Eval harness (B6):** endpoint + `EvalRunIn` + `eval_lab_enabled` flag exist but return a hardcoded pass.
* **Speculative decoding (B9):** all settings (`speculative_decoding_enabled`, `draft_model_path`, `cache_prompt`) + status reporting exist; only `_server_args` wiring is missing.
* **Structured outputs (A10) / prompt cache (A2):** capability probing (`capability_profile`) already distinguishes providers; the JSON prompts already define the shapes.
* **Sandbox (B10):** env allow-list + process tracking/reaping already establish the boundary to extend.

The disciplined edge of this codebase — SSRF pinning, path guards, off-by-default command/vision, redaction — is its real moat. Every item above treats **"no regression in the security boundary" as a hard acceptance criterion (§5)**, ideally machine-enforced by B6's eval checks. Ship in the §4 order and each release is independently valuable, independently revertible, and provably safe.

*End of roadmap.*

