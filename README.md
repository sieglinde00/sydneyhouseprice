# Sydney Housing Price Prediction

For a new Windows machine with Python:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-lock.txt
.\.venv\Scripts\python.exe train.py
.\.venv\Scripts\python.exe compare_valuations.py
.\.venv\Scripts\python.exe -m streamlit run app.py
```
Open [the local app](http://127.0.0.1:8501).
