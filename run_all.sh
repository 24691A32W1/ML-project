#!/usr/bin/env bash
# One-click: install -> generate data -> train -> launch dashboard
set -e
cd "$(dirname "$0")"
python -m pip install -r requirements.txt
python -m src.generate_dataset
python -m src.train_models
streamlit run app.py
