# ATS Resume Analyzer

A Resume ATS Analyzer built with:
- FastAPI backend
- Streamlit frontend

## Project Structure
- `backend/` → FastAPI API
- `app/` → Streamlit UI

## Run Locally

### Backend
```bash
cd backend
pip install -r requirements.txt
uvicorn main:app --reload --host 127.0.0.1 --port 8000

---

### Frontend

```bash
cd app
pip install -r requirements.txt
streamlit run streamlit_app.py
```
