import uvicorn
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "accent_fix"))

if __name__ == "__main__":
    uvicorn.run(
        "api.api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        app_dir=os.path.join(os.path.dirname(os.path.abspath(__file__)), "accent_fix")
    )