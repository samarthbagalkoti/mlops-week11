# service.py
import bentoml
from bentoml.io import JSON, Text

svc = bentoml.Service("w11d1_bento_hello")

# Optional: a lightweight ping endpoint (NOT /healthz)
@svc.api(input=Text(), output=Text(), route="/ping")
def ping(_: str = "ping"):
    return "pong"

# Example predict endpoint
@svc.api(input=JSON(), output=JSON(), route="/predict")
def predict(payload: dict):
    # simple echo to keep W11:D1 moving; replace with your model logic
    return {"ok": True, "echo": payload}

