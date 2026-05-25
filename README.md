# 🤖 AI-Driven Workforce Management System

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100%2B-009688?logo=fastapi&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B?logo=streamlit&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-F7931E?logo=scikit-learn&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-Containerised-2496ED?logo=docker&logoColor=white)

> An end-to-end enterprise-grade solution that automates task allocation and predicts employee attrition using NLP, machine learning, and a real-time analytics dashboard — fully containerised with Docker.

---

## 📌 Overview

Managing a large workforce manually is slow, error-prone, and reactive. This system solves that with three intelligent layers:

- **NLP Task Allocation** — automatically matches the right employee to the right project using TF-IDF vectorisation and cosine similarity (75%+ accuracy)
- **Attrition Prediction** — a trained ML model flags employees at risk of leaving before it happens, with burnout risk triggers to alert HR proactively
- **Real-Time Dashboard** — a Streamlit interface giving managers live visibility over 200+ active projects, workload distribution, and employee risk scores

---

## 🏗️ Project Structure

```
AI-Work-Force-Management/
│
├── SRC/
│   ├── app.py              ← FastAPI backend — REST API endpoints
│   ├── dashboard.py        ← Streamlit manager dashboard
│   └── nlp_engine.py       ← TF-IDF + Cosine Similarity skill-matching engine
│
├── Models/
│   └── *.pkl               ← Serialized trained models (attrition predictor, skill matcher)
│
├── Datasets/
│   └── *.csv               ← Training datasets and simulation data
│
├── Database/               ← Database schema and connection setup
│
├── Dockerfile              ← Container build instructions
├── .dockerignore           ← Files excluded from Docker build context
├── start.sh                ← Startup script (launches FastAPI + Streamlit)
├── updated_emp_dataset.csv ← Primary employee dataset
├── requirements.txt        ← Python dependencies
├── .gitignore
└── README.md
```

---

## ⚙️ How It Works

### 1. NLP Skill Matching Engine
Each employee has a skill profile. Each project has a set of required skills. Both are converted into TF-IDF vectors and scored against each other using cosine similarity. The highest-scoring available employee is automatically allocated to the project.

```
Employee Skills  → TF-IDF Vector ──┐
                                    ├─→ Cosine Similarity Score → Best Match
Project Requirements → TF-IDF Vector ──┘
```

### 2. Attrition Prediction Model
A supervised classifier trained on employee features — tenure, overtime, satisfaction indicators, workload — outputs a risk score per employee. If the score crosses a threshold, a burnout trigger fires.

```
Employee Features → ML Model (.pkl) → Risk Score → [Low | Medium | High]
                                                          ↓ (if High)
                                                    Risk-Aware Alert Triggered
```

### 3. FastAPI Backend
`app.py` exposes async REST endpoints for task allocation, attrition scoring, and employee data queries. Interactive Swagger docs auto-generated at `/docs`.

### 4. Streamlit Dashboard
`dashboard.py` gives managers a browser-based interface to monitor project allocations, employee risk scores, and workload distribution across 200+ active projects.

---

## 🛠️ Tech Stack

| Layer | Technology | Purpose |
|---|---|---|
| Backend API | FastAPI + Uvicorn | Async REST API server |
| Dashboard UI | Streamlit | Manager-facing web interface |
| ML Modelling | scikit-learn, SciPy | Attrition classifier, TF-IDF, cosine similarity |
| Data Processing | pandas, NumPy | Feature engineering, dataset manipulation |
| Model Persistence | joblib | Serialising/loading `.pkl` model files |
| Containerisation | Docker | Single-container deployment via `start.sh` |
| Utilities | python-multipart, requests, python-dotenv | File uploads, HTTP calls, env config |
| Language | Python 3.9+ | — |

---

## 🐳 Docker Deployment (Recommended)

The easiest way to run the entire system in one command.

### Prerequisites
- [Docker](https://docs.docker.com/get-docker/) installed and running

### 1. Clone the repository
```bash
git clone https://github.com/gokuunder2/AI-Work-Force-Management.git
cd AI-Work-Force-Management
```

### 2. Build the Docker image
```bash
docker build -t ai-workforce-management .
```

### 3. Run the container
```bash
docker run -p 8000:8000 -p 8501:8501 ai-workforce-management
```

### 4. Access the services
| Service | URL |
|---|---|
| FastAPI backend | `http://localhost:8000` |
| Swagger API docs | `http://localhost:8000/docs` |
| Streamlit dashboard | `http://localhost:8501` |

> The `start.sh` script inside the container starts both the FastAPI backend and Streamlit dashboard automatically.

---

## 💻 Manual Setup (Without Docker)

### 1. Clone the repository
```bash
git clone https://github.com/gokuunder2/AI-Work-Force-Management.git
cd AI-Work-Force-Management
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Start the FastAPI backend
```bash
uvicorn SRC.app:app --reload --host 0.0.0.0 --port 8000
```

### 4. Launch the Streamlit dashboard
```bash
streamlit run SRC/dashboard.py
```

---

## 📊 Dataset

| File | Description |
|---|---|
| `updated_emp_dataset.csv` | Primary employee records — used for model training and live allocation |
| `Datasets/*.csv` | Additional training sets and simulation scenarios |

Key employee fields include: `employee_id`, `department`, `skills`, `tenure`, `overtime_hours`, `satisfaction_score`, `attrition_label`.

---

## 🤖 ML Models

Pre-trained models live in `/Models` as `.pkl` files serialised via `joblib`. They load at API startup — no retraining required to run the system.

| Model | Type | Task |
|---|---|---|
| Attrition predictor | Supervised classifier | Predict employee churn risk |
| Skill matcher | TF-IDF + cosine similarity | Match employees to project requirements |

To retrain: update the datasets in `/Datasets`, re-run the training scripts, and re-export `.pkl` files to `/Models`.

---

## 📈 Key Metrics

| Metric | Value |
|---|---|
| Skill matching accuracy | **75%+** |
| Active projects monitored | **200+** |
| API type | Async (FastAPI) |
| Deployment | Docker containerised |

---

## 🔮 Future Improvements

- [ ] Role-based access control on the dashboard (manager vs HR vs admin views)
- [ ] Slack / email notification integration for burnout alerts
- [ ] Upgrade NLP engine to sentence transformers for deeper semantic skill matching
- [ ] Add collaborative filtering based on past project performance
- [ ] CI/CD pipeline with GitHub Actions

---

## 🤝 Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you'd like to change.

---

## 📄 License

This project is for educational and portfolio purposes.
