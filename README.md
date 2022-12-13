# FRE7773finalproject
## Description
## Getting Started
### Code and Data
1. Project can be reached at https://github.com/Innocence0619/Breastcancer
2. Raw data can be download at https://www.kaggle.com/datasets/yasserh/breast-cancer-dataset
### Executing Project
1. Check all the packages are installed by code \n
`pip install -r requirements.txt`
2. Run bc_flow.py to create a metaflow data \n
`python3 bc_flow.py run --with card`
3. Make sure you have run the python file and then run app.py to generate a webpage for predicting the breast cancer type
`python3 app.py run`
4. Check the `breast cancer.ipynb` file to see all the EDA and whole process 
### Files instructions
1. app.py - store the flask source code
2. bc_flow.py - generate a metaflow history with DAG
3. breast-cancer.csv - store the origin data
4. breastcancer.ipynb - original playground with data
5. requirements1.txt - packages that are needed
