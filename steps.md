# Run the app

1. Open a terminal in this project folder.
2. Create and activate a virtual environment (recommended):
   - Windows (Command Prompt): `python -m venv venv` then `venv\Scripts\activate`
   - macOS / Linux: `python3 -m venv venv` then `source venv/bin/activate`
3. Install the dependencies: `pip install -r requirements.txt`
4. Start the app: `streamlit run app.py`
5. Open the local address Streamlit prints (default http://localhost:8501).

Notes:
- The first start downloads the pretrained speech model from Hugging Face, so
  Speech mode needs an internet connection the first time it loads.
- Face mode needs the model file `model_78.h5` in this folder. That file is not
  in the repository — see "Known limitations" in README.md.
- Create an account on the signup screen first; accounts are stored locally in
  `users.json` (passwords hashed with bcrypt; the file is git-ignored).
