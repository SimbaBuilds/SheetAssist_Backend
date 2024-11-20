import sys
import os
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.endpoints import process_query
import uvicorn

app = FastAPI()

#FASTAPI
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Specify domains if needed
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


#include endpoints via router
#region
app.include_router(process_query.router)
#endregion


# development
# file_paths = ['course_data.csv']
# data_info_list = []

# # Read the CSV files
# for file_path in file_paths:
#     df = pd.read_csv(file_path)
#     # Store data info in a DataInfo object
#     data_info = FileDataInfo(
#         content=df, 
#         snapshot=str(df.head(10)), 
#         data_type="DataFrame", 
#         original_file_name=file_path
#     )
#     data_info_list.append(data_info)

# query = "Remove courses with less than 20 active students from this list."
    

# Example usage
if __name__ == "__main__":    
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="localhost", port=port)




