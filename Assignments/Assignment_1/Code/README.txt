This folder contains the template code for a search engine application. 

main.py - The main module that contains the outline of the Search Engine. Do not change anything in this file.
util.py - An extra file where you can add any additional processing or utility functions that you may need for any of the sub-tasks.
sentenceSegmentation.py, tokenization.py, inflectionReduction.py and stopwordRemoval.py - Implement the corresponding sub-tasks inside the functions in these files.

More files corresponding to each sub-task will be provided as the assignment progresses, along with updated versions of main.py

To test your code, run main.py with the appropriate arguments
Usage: main.py [-custom] [-dataset DATASET FOLDER] [-out_folder OUTPUT FOLDER]
               [-segmenter SEGMENTER TYPE (naive|punkt)] [-tokenizer TOKENIZER TYPE (naive|ptb)] 
When the -custom flag is passed, the system will take a query from the user as input. When the flag is not passed, all the queries in the Cranfield dataset are considered, for example:
> python main.py -custom
> Enter query below
> Papers on Aerodynamics
This will generate *queries.txt files in the OUTPUT FOLDER after each stage of preprocessing of the query and *docs.txt files in the OUTPUT FOLDER after each stage of preprocessing of the documents.
