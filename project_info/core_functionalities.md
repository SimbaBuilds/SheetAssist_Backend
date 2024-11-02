#Core Backend Functionalities
1. Receive and pre-process form data from the front end via endpoint "process query"
    1. Web URLs (up to 10): Links to Google Sheets or Excel for Web Sheets 
        1. Tabular data from web URLs must be read and converted to csvs for processing
    2. Files (up to 10) of type .xlsx, .csv, .json, .docx, .txt, ,pdf, .jpeg, and .png, 
        1. .xlsx files must be converted to csv files
        2. json data must be loaded and converted to a string
        3. .docx and .txt files must be loaded and converted to a string
        4. .png files must be converted to .jpeg
    3. User query: the desired action expressed by the user
2. Once preprocessing is done, further processing can begin
    1. A sandbox environment must be set up as implemented in my EnhancedPythonInterpreter class
    2. Snapshots of the data along with the user query are sent to LLMs which iteratively generate code that 
        results in the user query being satsified
    3. A response to the front end is sent which will always be one or more pandas dataframes or an error/feedback message
