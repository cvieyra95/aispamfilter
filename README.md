# Spam Email Classifier

## System Requirements
* Operating System: Windows 10/11, MacOS
* Language: Python 3.11 (or higher)
* Libraries: 
    - FastAPI
    - Scikit-learn
    - Joblib
    - Pydantic

## Installation Instructions

1. Install Python
    - Ensure Python 3.11 or higher is intalled.
        - `python --version`
    - Python can be dowloaded from `https://www.python.org/downloads/`

2. Install Dependencies
   - Run this command in the project root folder(the same folder containing api.py)
   - `pip install fastapi uvicorn scikit-learn joblib pandas numpy pydantic`
   - (We trained the model using scitkit-learn version 1.7.2)

## Running the application
    
### Method 1 (Using Two Terminals)

1. Open two terminal windows

2. Change to the project directory on both terminal windows:

3. On one terminal window run the FastAPI(Backend)
    - From the project root folder (the same folder containing api.py)
       - run: `uvicorn api:app --reload`

4. On the second terminal window 
    - change to the frontend folder `cd frontend`
    - Run:
        `python -m http.server 5500`
    - You can now go to `http://localhost:5500' to the webpage and use the application

### Method 2 (Through VSCode)

1. Open project in VSCode

2. Install the Live Server extension in VSCode by Ritwick Dey

3. Open a terminal in vscode and start the api
    - From the project root folder (the same folder containing api.py)
       - run: `uvicorn api:app --reload`

4. Go to frontend folder and right click on `index.hmtl` and click on open with live server
    - webpage should open with default browser
    



 
