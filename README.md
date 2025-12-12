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
2. Open two terminal windows and change directory to project folder

3. Create a Virtual Environment (recommened)
    - Windows
    `venv\Scripts\activate`
    - macOS
    `source venv/bin/activate`
4. Install Dependencies
   - Run the following command in the project where api.py is located
   `pip install uvicorn scikit-learn pandas numpy`

   
5. Running the application
    ### Backend
    From the project root folder, run:
    `uvicorn api:app -reload`

    ### Frontend
    Run:
    `python -m http.server 5500`

6. Using Application
    - Once the backend and frontend are running you can now use the application,
    - Paste the contents of an email into the entry box and the application will tell if you its spam or not. 
    



 
