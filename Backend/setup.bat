echo Creating virtual environment and installing dependencies.
python -m venv venv
".\venv\Scripts\"activate & pip install -r requirements.txt & deactivate
echo Installation finished.
