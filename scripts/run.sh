#!/bin/bash
cd "$(dirname "$0")"
echo "========================================================="
echo " ASTM E155 Radiographic Testing System Launcher "
echo "========================================================="
echo "Installing/verifying python dependencies..."
pip install -r requirements.txt
echo "Launching Streamlit Web App on http://localhost:8502 ..."
streamlit run app.py --server.port 8502
