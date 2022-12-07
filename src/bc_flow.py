import os
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from metaflow import FlowSpec, IncludeFile, Parameter, current, step

from comet_ml import Experiment
# make sure we are running locally for this
assert os.environ.get('METAFLOW_DEFAULT_DATASTORE', 'local') == 'local'
assert os.environ.get('METAFLOW_DEFAULT_ENVIRONMENT', 'local') == 'local'
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import seaborn as sns
from sklearn.metrics import f1_score, precision_score, accuracy_score, recall_score, confusion_matrix
class BreastCancerFlow(FlowSpec):
    """
    BreastCancerFlow is a minimal DAG using several machine learning models to predict the breast cancer type.
    """

    DATA_FILE = IncludeFile(
        'dataset',
        help='The dataset from kaggle',
        default='breast-cancer.csv')

    TEST_SPLIT = Parameter(
        name='test_split',
        help='Determining the split of the dataset for testing',
        default=0.30
    )

    @step
    def start(self):
        """
        Start up and print out some info to make sure everything is ok metaflow-side
        """
        print("Starting up at {}".format(datetime.utcnow()))
        # debug printing - this is from https://docs.metaflow.org/metaflow/tagging
        # to show how information about the current run can be accessed programmatically
        print("flow name: %s" % current.flow_name)
        print("run id: %s" % current.run_id)
        print("username: %s" % current.username)
        self.next(self.load_data)

    @step
    def load_data(self): 
        """
        Read the data in from the breast-cancer file
        """
        from io import StringIO

        import pandas as pd

        df = pd.read_csv(StringIO(self.DATA_FILE))
        df = df.set_index(['id'])
        self.dataset = df
        self.target = "diagnosis"
        df.isna().sum()
        #may have to define different species after
        # female = self.dataset[self.dataset["Gender_Male"]==0]
        # male = self.dataset[self.dataset["Gender_Male"]==1]
        self.Xs = self.dataset.drop(['diagnosis'],axis =1)
        self.Ys = self.dataset['diagnosis']
        self.Ys = pd.DataFrame([0 if i=='B' else 1 for i in self.Ys], columns= ["y"])
        dataset2 = self.dataset[(self.dataset['radius_mean']<=15.3) & (self.dataset['radius_mean']>=13)]
        self.index1 = df.index
        self.index2 = list(dataset2.index)
        print(self.index2)
        # self.Xfe = female.drop(['Loan_Status','Loan_ID'],axis =1)
        # self.Yfe = female['Loan_Status']
        # self.Xma = male.drop(['Loan_Status','Loan_ID'],axis =1)
        # self.Yma = male['Loan_Status']
        # self.Ys = LabelEncoder().fit_transform(self.dataset[self.target])
        # self.Xs = self.dataset.drop(columns=ignore_features)
        self.next(self.check_dataset)

    @step
    def check_dataset(self):
        """
        Check data is ok before training starts
        """
        self.next(self.prepare_data)

    @step 
    def prepare_data(self):
        """
        Using Standard Scalar and PCA to nomralize and reduce the dimentionality of X
        """
        self.standard = StandardScaler()
        self.standard.fit(self.Xs)
        self.X_std = StandardScaler().fit_transform(self.Xs)
        #the higher PCA parameter is, the more features we will have.
        self.modelpca = PCA(0.95)
        self.modelpca.fit(self.X_std)
        self.X_pca = pd.DataFrame(self.modelpca.fit_transform(self.X_std))
        self.X_pca['id'] = self.index1
        self.index2 = self.index2
        # cumsum = np.cumsum(pca.explained_variance_ratio_)
        # plt.plot(cumsum)
        # pca.explained_variance_ratio_
        self.next(self.prepare_train_and_test_dataset)

    @step
    def prepare_train_and_test_dataset(self):
        """
        Split traning dataset and testing dataset
        """
        self.standard = self.standard
        self.modelpca = self.modelpca
        self.index2 = self.index2
        from sklearn.model_selection import train_test_split

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X_pca, 
            self.Ys, 
            test_size=self.TEST_SPLIT, 
            random_state=42
            )
        self.index_test = pd.DataFrame(self.X_test['id'])
        self.X_train = self.X_train.iloc[:,:-1]
        self.X_test = self.X_test.iloc[:,:-1]
        self.next(self.logistic_regression, self.decision_tree_model, self.random_forest_model, self.naive_bayes, self.knn)
    
    @step
    def logistic_regression(self):
        """
        Create Logistic Regression Model to train data 
        """
        self.standard = self.standard
        self.modelpca = self.modelpca
        self.index2 = self.index2
        self.index_test = self.index_test
        logistic = LogisticRegression()
        self.cross_val_scores1 = cross_val_score(logistic, self.X_train, self.y_train.values.ravel(), cv=5,scoring='recall').mean()
        logistic.fit(self.X_train, self.y_train.values.ravel())
        # versioned the trained model using self
        self.logistic= logistic
        # go to the testing phase
        self.next(self.join)

    @step
    def decision_tree_model(self):
        """
        Create Decision Tree Model to train data 
        """
        decision_tree = DecisionTreeClassifier()
        self.cross_val_scores2 = cross_val_score(decision_tree, self.X_train, self.y_train.values.ravel(), cv=5,scoring='recall').mean()
        decision_tree.fit(self.X_train, self.y_train.values.ravel())
        # versioned the trained model using self
        self.decistion_tree= decision_tree
        # go to the testing phase
        self.next(self.join)

    @step
    def random_forest_model(self):
        """
        Create Random Forest Model to train data 
        """
        random_forest = RandomForestClassifier()
        self.cross_val_scores3 = cross_val_score(random_forest, self.X_train, self.y_train.values.ravel(), cv=5,scoring='recall').mean()
        random_forest.fit(self.X_train, self.y_train.values.ravel())
        # versioned the trained model using self
        self.random_forest= random_forest
        # go to the testing phase
        self.next(self.join)

    @step
    def naive_bayes(self):
        """
        Create Naive Bayes Model to train data 
        """
        Naive_Bayes = GaussianNB()
        self.cross_val_scores4 = cross_val_score(Naive_Bayes, self.X_train, self.y_train.values.ravel(), cv=5,scoring='recall').mean()
        Naive_Bayes.fit(self.X_train, self.y_train.values.ravel())
        # versioned the trained model using self
        self.Naive_Bayes= Naive_Bayes
        # go to the testing phase
        self.next(self.join)
    
    @step
    def knn(self):
        """
        Create KNN Model to train data 
        """
        KNN = KNeighborsClassifier(n_neighbors=12)
        self.cross_val_scores5 = cross_val_score(KNN, self.X_train, self.y_train.values.ravel(), cv=5,scoring='recall').mean()
        KNN.fit(self.X_train, self.y_train.values.ravel())
        # versioned the trained model using self
        self.KNN= KNN
        # go to the testing phase
        self.next(self.join)

    @step
    def join(self, inputs):
        """
        Take all metrics together and find the best model
        """
        self.standard =inputs.logistic_regression.standard
        self.modelpca =inputs.logistic_regression.modelpca
        self.index2 = inputs.logistic_regression.index2
        self.index_test = inputs.logistic_regression.index_test
        self.X_train = inputs[0].X_train
        self.y_train = inputs[0].y_train
        self.X_test = inputs[0].X_test
        self.y_test = inputs[0].y_test 
        # test the models with accuracy score
        # Create an experiment with your api key
        predictions1 = inputs.logistic_regression.logistic.predict(inputs.logistic_regression.X_test)
        self.accuracy_score1 = accuracy_score(inputs.logistic_regression.y_test, predictions1)
        self.f11 = f1_score(inputs.logistic_regression.y_test, predictions1)
        self.precision1 = precision_score(inputs.logistic_regression.y_test, predictions1)
        self.recall1 = recall_score(inputs.logistic_regression.y_test, predictions1)
        self.cross_val_scores1 = inputs.logistic_regression.cross_val_scores1
        sns.set()
        f1, ax1 = plt.subplots()
        C1 = confusion_matrix(inputs.logistic_regression.y_test, predictions1)
        print(C1)
        sns.heatmap(C1, annot=True, ax=ax1)
        ax1.set_title('Logistic_regression confusion matrix')
        ax1.set_xlabel('predict')
        ax1.set_ylabel('true')
        plt.show()
        predictions2 = inputs.decision_tree_model.decistion_tree.predict(inputs.decision_tree_model.X_test)
        self.accuracy_score2 = accuracy_score(inputs.decision_tree_model.y_test, predictions2)
        self.f12 = f1_score(inputs.decision_tree_model.y_test, predictions2)
        self.precision2 = precision_score(inputs.decision_tree_model.y_test, predictions2)
        self.recall2 = recall_score(inputs.decision_tree_model.y_test, predictions2)
        self.cross_val_scores2 = inputs.decision_tree_model.cross_val_scores2
        sns.set()
        f2, ax2 = plt.subplots()
        C2 = confusion_matrix(inputs.decision_tree_model.y_test, predictions2)
        print(C2)
        sns.heatmap(C2, annot=True, ax=ax2)
        ax2.set_title('Decision_tree_model confusion matrix')
        ax2.set_xlabel('predict')
        ax2.set_ylabel('true')
        plt.show()

        predictions3 = inputs.random_forest_model.random_forest.predict(inputs.random_forest_model.X_test)
        self.accuracy_score3 = accuracy_score(inputs.random_forest_model.y_test, predictions3)
        self.f13 = f1_score(inputs.random_forest_model.y_test, predictions3)
        self.precision3 = precision_score(inputs.random_forest_model.y_test, predictions3)
        self.recall3 = recall_score(inputs.random_forest_model.y_test, predictions3)
        self.cross_val_scores3 = inputs.random_forest_model.cross_val_scores3
        sns.set()
        f3, ax3 = plt.subplots()
        C3 = confusion_matrix(inputs.random_forest_model.y_test, predictions3)
        print(C3)
        sns.heatmap(C3, annot=True, ax=ax3)
        ax3.set_title('Random_forest_model confusion matrix')
        ax3.set_xlabel('predict')
        ax3.set_ylabel('true')
        plt.show()

        predictions4 = inputs.naive_bayes.Naive_Bayes.predict(inputs.naive_bayes.X_test)
        self.accuracy_score4 = accuracy_score(inputs.naive_bayes.y_test, predictions4)
        self.f14 = f1_score(inputs.naive_bayes.y_test, predictions4)
        self.precision4 = precision_score(inputs.naive_bayes.y_test, predictions4)
        self.recall4 = recall_score(inputs.naive_bayes.y_test, predictions4)
        self.cross_val_scores4 = inputs.naive_bayes.cross_val_scores4
        sns.set()
        f4, ax4 = plt.subplots()
        C4 = confusion_matrix(inputs.naive_bayes.y_test, predictions4)
        print(C4)
        sns.heatmap(C4, annot=True, ax=ax4)
        ax4.set_title('Naive_bayes confusion matrix')
        ax4.set_xlabel('predict')
        ax4.set_ylabel('true')
        plt.show()

        predictions5 = inputs.knn.KNN.predict(inputs.knn.X_test)
        self.accuracy_score5 = accuracy_score(inputs.knn.y_test, predictions5)
        self.f15 = f1_score(inputs.knn.y_test, predictions5)
        self.precision5 = precision_score(inputs.knn.y_test, predictions5)
        self.recall5 = recall_score(inputs.knn.y_test, predictions5)
        self.cross_val_scores5 = inputs.knn.cross_val_scores5
        sns.set()
        f5, ax5 = plt.subplots()
        C5 = confusion_matrix(inputs.knn.y_test, predictions5)
        print(C5)
        sns.heatmap(C5, annot=True, ax=ax5)
        ax5.set_title('Knn confusion matrix')
        ax5.set_xlabel('predict')
        ax5.set_ylabel('true')
        plt.show()
        experiment = Experiment(
        api_key="VMoZEQdFLMMdOCaRLTDE64VQP",
        project_name="fre7773finalproject",
        workspace="nyu-fre-7773-2021",
        )
        self.accuracy_scores = [self.accuracy_score1, self.accuracy_score2, self.accuracy_score3, self.accuracy_score4, self.accuracy_score5]
        self.f1s = [self.f11, self.f12, self.f13, self.f14, self.f15]
        self.precisions = [self.precision1,self.precision2,self.precision3,self.precision4,self.precision5]
        self.recalls = [self.recall1,self.recall2,self.recall3,self.recall4,self.recall5]
        self.cross_val_scoress = [self.cross_val_scores1,self.cross_val_scores2,self.cross_val_scores3,self.cross_val_scores4,self.cross_val_scores5]
        self.params_list =[
                  'Logistic Regression',
                  'Decision Tree',
                  'Random Forest',
                  'Naive Bayes',
                  'KNN'
        ]
        for i, model in enumerate(self.params_list):
            experiment.log_parameter(f"model", model)
            metrics = {
                f"accuracy_scores_{model}": self.accuracy_scores[i],
                f"f1{model}": self.f1s[i],
                f"precisions_{model}": self.precisions[i],
                f"recalls_{model}": self.recalls[i],
                f"cross_val_scores_{model}": self.cross_val_scoress[i],
            }
            experiment.log_metrics(metrics)
        self.next(self.grid_search)

    @step
    def grid_search(self):
        self.standard = self.standard
        self.modelpca = self.modelpca
        self.index2 = self.index2
        self.index_test = self.index_test
        """
        After finding the model, use grid search to do hyper-parameter tuning.
        """
        param_grid = {"max_iter": [80, 100, 120],
                      "fit_intercept": [True, False],
                      "multi_class": ['ovr', 'multinomial', 'auto']}
        logistic = LogisticRegression()
        grid_search = GridSearchCV(
            logistic, param_grid, cv=5, scoring="recall", return_train_score=True
        )
        grid_search.fit(self.X_train, self.y_train.values.ravel())
        print("The best recall is",grid_search.best_score_)
        print("Best parameters:", '\n', grid_search.best_params_)
        model1 = LogisticRegression(fit_intercept= False, max_iter= 80, multi_class= 'ovr')
        model1.fit(self.X_train, self.y_train.values.ravel())
        self.y_predicted = model1.predict(self.X_test)
        print(self.index_test['id'])
        self.next(self.test_model)

    @step
    def test_model(self):
        """
        Test our best model on the test data
        """
        self.standard = self.standard
        self.modelpca = self.modelpca
        self.model = LogisticRegression(fit_intercept= False, max_iter= 80, multi_class= 'ovr')
        print(self.model)
        self.model.fit(self.X_train, self.y_train)
        predictions = self.model.predict(self.X_test)
        testxy = pd.DataFrame(self.X_test)
        testxy['y'] = self.y_test
        testxy['predictions'] = predictions
        testxy['id']=list(self.index_test['id'].values)
        p1 = testxy[testxy['id'].isin(self.index2)]
        print(testxy['id'])
        p2 = testxy[~testxy['id'].isin(self.index2)]
        self.recall_less = recall_score(p1['y'], p1['predictions'])
        self.recall_greater = recall_score(p2['y'], p2['predictions'])
        print(self.recall_less,self.recall_greater)
        cross_val_scores = cross_val_score(self.model, self.X_train, self.y_train.values.ravel(), cv=5)
        accuracy = accuracy_score(self.y_test, predictions)
        f1 = f1_score(self.y_test, predictions)
        precision = precision_score(self.y_test, predictions)
        recall = recall_score(self.y_test, predictions)
        '''
        sns.set()
        f6, ax6 = plt.subplots()
        C6 = confusion_matrix(self.y_test, predictions)
        print(C6)
        sns.heatmap(C6, annot=True, ax=ax6)
        ax6.set_title('Logistic Regression confusion matrix with optimal parameters')
        ax6.set_xlabel('predict')
        ax6.set_ylabel('true')
        plt.show()

        f7, ax7 = plt.subplots()
        C7 = confusion_matrix(p1['y'], p1['predictions'])
        print(C7)
        sns.heatmap(C7, annot=True, ax=ax7)
        ax7.set_title('Logistic Regression confusion matrix with radius_mean[13,15.3]')
        ax7.set_xlabel('predict')
        ax7.set_ylabel('true')
        plt.show()

        f8, ax8 = plt.subplots()
        C8 = confusion_matrix(p2['y'], p2['predictions'])
        print(C8)
        sns.heatmap(C8, annot=True, ax=ax8)
        ax8.set_title('Logistic Regression confusion matrix with less or greater radius_mean')
        ax8.set_xlabel('predict')
        ax8.set_ylabel('true')
        plt.show()
        '''
        print('accuracy_score is {}, f1_score is {}, precision_score is {}, recall_score is {}, cross_val_scores is {}'.format(accuracy, f1, precision, recall, cross_val_scores))
        # all is done go to the end
        self.next(self.end)


    @step
    def end(self):
        """
        It's the end
        """
        # all done, just print goodbye
        print("All done at {}!\n See you, space cowboys!".format(datetime.utcnow()))

if __name__ == '__main__':
    BreastCancerFlow()
