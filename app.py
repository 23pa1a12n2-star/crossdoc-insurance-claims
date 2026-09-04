import uvicorn
from fastapi import FastAPI
from fastapi.responses import HTMLResponse

app = FastAPI(title="CrossDoc Insurance Evidence Review")

@app.get("/", response_class=HTMLResponse)
async def serve_ui():
    # Will be connected to Jinja2 in Phase 6
    return "<h1>CrossDoc Engine Starting...</h1>"

if __name__ == "__main__":
    # Required exactly for `python app.py` startup
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
