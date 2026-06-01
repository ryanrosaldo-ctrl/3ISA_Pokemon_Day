#!/bin/bash
echo "============================================"
echo " POKEMON DAY II - 3ISA ENGINE"
echo "============================================"
echo "Installing dependencies..."
pip install -r requirements.txt
echo ""
echo "Starting app... Open: http://localhost:8501"
streamlit run app.py
