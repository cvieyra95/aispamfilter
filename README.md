# Spam Email Classifier

## Instructions

### System Requirements
* Operating System: Windows 10/11, MacOS
* Language: Python 3.11 (or higher)
* Libraries: 
    - FastAPI
    - Scikit-learn
    - Joblib
    - Pydantic

## Installation Instructions

1. Install Python
    Ensure Python 3.11 or higher is intalled.
    `python --version`
2. Open Project Folder In VSCode

3. Create a Virtual Environment (recommened)
    Windows
    `venv\Scripts\activate`
    macOS
    `source venv/bin/activate`
4. Install Dependencies
   Run the following command in the project where api.py is located
   `pip install -r requirements.txt`

   Install the VSCode liver server extension

5. Running the application
    ### Backend
    From the project root folder, run:
    `uvicorn api:app -reload`

    ### Frontend
    Open the index.html file in VSCode live server(right click then open with live server)



 
