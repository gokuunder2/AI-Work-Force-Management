#!/bin/bash
# Start FastAPI in the background
uvicorn SRC.app:app --host 0.0.0.0 --port 8000 &

# Start Streamlit in the foreground
streamlit run SRC/dashboard.py --server.port 5000 --server.address 0.0.0.0