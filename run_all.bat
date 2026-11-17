@echo off
cd /d %~dp0
python -m pip install -r requirements.txt
python -m src.generate_dataset
python -m src.train_models
streamlit run app.py
