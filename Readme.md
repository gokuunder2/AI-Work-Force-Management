# 🤖 AI-Driven Workforce Management System

An end-to-end enterprise solution that optimizes human capital management using a **FastAPI architecture**, **NLP-based task allocation**, and **Machine Learning attrition prediction**.

## 🏗️ Project Structure
The project is organized into a modular structure for scalability:
*   **`/SRC`**: Contains the core logic (`app.py`, `dashboard.py`, and the NLP engine).
*   **`/Models`**: Serialized ML models (`.pkl` files) for attrition and skill matching.
*   **`/Datasets`**: CSV files used for training and simulation.

## 🛠️ Key Technical Features
*   **NLP Matching Engine:** Uses TF-IDF Vectorization and Cosine Similarity to align employee skills with project requirements.
*   **Predictive Analytics:** Features a machine learning model to predict employee attrition and prevent burnout.
*   **Real-time Dashboard:** A Streamlit-based interface for managers to visualize task allocations and risk metrics.

## 🚀 Quick Start
1. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
2. **Run the Backemd:**
   ```bash
   pip install -r requirements.txt
2. **Launch the UI:**
   ```bash
   streamlit run SRC.dashboard.py

      