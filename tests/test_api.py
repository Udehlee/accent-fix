from fastapi.testclient import TestClient
from accent_fix.api.api import app

client = TestClient(app)

def test_home_route():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "AccentFix is running"}

def test_upload_audio_no_file():
    response = client.post("/upload-audio")
    assert response.status_code == 422  # validation error — no file sent

def test_upload_audio_with_file():
    with open("tests/sample_audio.mp3", "rb") as f:
        response = client.post(
            "/upload-audio",
            files={"file": ("sample_audio.mp3", f, "audio/mpeg")}
        )
    assert response.status_code == 200
    data = response.json()
    assert "accent" in data
    assert "original_transcript" in data
    assert "corrected_transcript" in data
    assert "summary" in data