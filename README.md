TRACK_ID=PS02

# CrossDoc: Insurance Claims Evidence Review Assistant

CrossDoc is an evidence-first claims review engine designed specifically for motor insurance scenarios. It deterministically surfaces cross-document contradictions and verifies claims against policy vectors, ensuring that zero-shot LLM reasoning never independently denies or approves a claim.

## Setup Instructions

This application features zero internal build steps and starts instantly using an embedded NumPy semantic index to guarantee cross-OS compatibility without C++ compiler issues.

```bash
pip install -r requirements.txt
python app.py
```

The application will bind to port 8000. Open `http://localhost:8000` in your browser.

## Architecture

* **LLM Engine**: `gemini-3.5-flash-lite` (via google-genai)
* **Embedding Model**: `gemini-embedding-001`
* **Local Backend**: FastAPI + Uvicorn + Pydantic
* **State Engine**: Pure Python CrossDoc matrix
* **Vector DB**: Pre-compiled Python NumPy `policy_index.json` (0 dependency setup)

## Demonstration Data

This repository contains completely synthetic data (`data/cases/`) to evaluate the evaluation engine handling clear approvals, date contradictions, missing required documentation, and semantic policy exclusions (e.g. racing).
