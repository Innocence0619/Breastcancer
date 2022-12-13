# FRE7773finalproject
## Description
## Getting Started
### Code and Data
1. Project can be reached at https://github.com/Innocence0619/Breastcancer
2. Raw data can be download at https://www.kaggle.com/datasets/yasserh/breast-cancer-dataset
### Executing Project
1. Check all the packages are installed by code.

`pip install -r requirements.txt`

2. Run bc_flow.py to create a metaflow data.

`python3 bc_flow.py run --with card`

3. Make sure you have run the python file and then run app.py to generate a webpage for predicting the breast cancer type.

`python3 app.py run`

4. Check the `breast cancer.ipynb` file to see all the EDA and whole process.

## Files instructions
1. app.py - store the flask source code
2. bc_flow.py - generate a metaflow history with DAG
3. breast-cancer.csv - store the origin data
4. breastcancer.ipynb - original playground with data
5. requirements1.txt - packages that are needed

## Deployment
1. Using AWS EC2 
Apart from using html, we created an EC2 instance for our prediction
<img width="924" alt="Screenshot_20221212_090210" src="https://user-images.githubusercontent.com/53091204/207222080-f7239caf-ca84-46e2-aad6-1aab7c005edc.png">
2. Comet playground

We have updated metrics needed into Comet 
Visit by :https://www.comet.com/nyu-fre-7773-2021/fre7773finalproject?shareable=JMrntmCQVqDBPHKJ5Nl49U2X0

<img width="910" alt="Screenshot_20221212_095736" src="https://user-images.githubusercontent.com/53091204/207223101-a548db88-cbcc-4a63-b940-66bcc30172b8.png">


