from google.cloud import aiplatform

PROJECT  = "pill-identify-466807"
LOCATION = "us-central1"

client = aiplatform.gapic.PredictionServiceClient(
    client_options={"api_endpoint": f"{LOCATION}-aiplatform.googleapis.com"}
)
endpoint = (f"projects/{PROJECT}/locations/{LOCATION}"
            "/publishers/google/models/gemini-embedding-001")

resp = client.predict(
    endpoint=endpoint,
    instances=[{"content": "테스트 문장"}]
)
print(len(resp.predictions[0]), "dims")   # → 3072 나오면 성공
