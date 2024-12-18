## Data Visualization Feature


1. I am implementing a data visualization feature to the project.  Form data from the front end will be (1)url or file (2) color palette selection (3) Optional custom instructions.

2. The visualization will be created via LLM generated code running in a sandbox environment similar to the way data processing is done in the main project feature.  In the existing sandbox implementation, the last line of code, either exp or assignment, is captured (@sandbox.py).

3. The LLM should get access to the user's custom instructions if any as well as a df snapshot -- snapshot should include the first few rows and some random rows

4. Instructions for LLM
  1. You can choose to produce one plot or two subplots depending on how you think the data should be visualized 
  2. Always remove grid lines.  Use axes[].grid(False) to remove all grid lines
  3. Use axes[].tick_params(axis='x', rotation=45) to make x axis labels on all plots and sub plots legible
  4. If creating subplots use plt.tight_layout() 


5. Once the visualization is created, by the LLM, we will save it to bytes buffer, use plt.savegif, buf.seek(0), plt.close, return buf.

6. The plot will be sent to the front end.  The user will have the option to download the file -- if download, the download endpoint in this codebase will be called.

  