import pandas as pd 
import numpy as np 

#Reading the excel file 

import pandas as pd

# Load answers
answers = pd.read_csv(r"C:\Users\Mudassir\Desktop\Edvancer Assignment Submission\Python 2\Assignment 1\Answers.csv", encoding='ISO-8859-1')

# Load tags
tags = pd.read_csv(r"C:\Users\Mudassir\Desktop\Edvancer Assignment Submission\Python 2\Assignment 1\Tags.csv", encoding='ISO-8859-1')

# Load questions
questions = pd.read_csv(r"C:\Users\Mudassir\Desktop\Edvancer Assignment Submission\Python 2\Assignment 1\Questions.csv", encoding='ISO-8859-1')



answers.head(20)

tags.head(20)

questions.head(20)


answers.info()

questions.info()

tags.info()


