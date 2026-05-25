FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .

# Use --no-cache-dir to ensure a fresh installation
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000
EXPOSE 5000

CMD python -m uvicorn SRC.app:app --host 0.0.0.0 --port 8000 & python -m streamlit run SRC/dashboard.py --server.port 5000 --server.address 0.0.0.0