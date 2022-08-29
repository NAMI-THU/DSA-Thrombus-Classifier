echo Creating virtual environment and installing dependencies.
python -m venv venv
".\venv\Scripts\"activate & pip install -r requirements.txt & pip install torch torchvision --extra-index-url https://download.pytorch.org/whl/cu116 & deactivate
echo Installation finished.
