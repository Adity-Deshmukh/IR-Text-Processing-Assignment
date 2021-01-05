# Information Retrieval Assignment 
# Topic : Text Processing
# Group Number : 11

-----------------------------------------------------------

The code is divided between two files:

- build_index.py
Usage: python build_index.py <path to corpus root>
Processes files in the subfolder 'AA' in the folder path provided, and builds the index using the documents in those files.
The inverted index is written to disk in the working directory with the filename 'index_new.pkl'.

- test_queries.py
Usage: python test_queries.py <path to index file>
Reads the index and processes queries using appropriate configuration as specified by the user.

Alternatively, the code can also be viewed using the HTML file provided in a notebook-like non-interactive format.

-----------------------------------------------------------

The project directory includes a single file from the corpus, using which index construction can be tested. (corpus/AA/wiki_00)

A pre-built index built on the same file is also included. (index_00.pkl)

The results reported have been obtained using a larger index built using the subset of the corpus in the AA folder.
Because of the large size of this file (542 MB), it has not been included in the submission, but can be accessed and downloaded from https://drive.google.com/drive/folders/1_vUERlNAmbK8XFOs8pf2C7xAJ9Dv_mqN?usp=sharing.