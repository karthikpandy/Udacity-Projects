#!/usr/bin/python

import sys
import pickle
sys.path.append("../tools/")
import matplotlib.pyplot
from time import time
from operator import itemgetter
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
import tester

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data, test_classifier
from sklearn.feature_selection import SelectKBest,SelectPercentile
import pandas as pd
from sklearn import preprocessing
from sklearn import cross_validation
from sklearn.pipeline import Pipeline


### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

print 'Number of records :',  len(data_dict.keys())

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_list = ['poi','salary', 'deferral_payments', 'total_payments', 'loan_advances', 'bonus', 'restricted_stock_deferred', 'deferred_income', 'total_stock_value', 'expenses', 'exercised_stock_options', 'other', 'long_term_incentive', 'restricted_stock', 'director_fees',
                'to_messages', 'from_poi_to_this_person', 'from_messages', 'from_this_person_to_poi', 'shared_receipt_with_poi'] 

j = 0
for i in features_list:
    print j, i
    j = j+1

### Task 2: Remove outliers

#Plot to see Outliers

features = ["salary", "bonus"]
data = featureFormat(data_dict, features)

for point in data:
    salary = point[0]
    bonus = point[1]
    #print(salary)
    matplotlib.pyplot.scatter( salary, bonus )

print('Salary :')
for i in  data_dict:
    if data_dict[i]['salary'] > 800000:
        if data_dict[i]['salary'] == 'NaN':
            pass
        else:
            print (i)

print('Bonus :')
for i in  data_dict:            
    if data_dict[i]['bonus'] > 4000000:
        if data_dict[i]['bonus'] == 'NaN':
            pass
        else:
            print(i)

matplotlib.pyplot.xlabel("salary")
matplotlib.pyplot.ylabel("bonus")
matplotlib.pyplot.title("Before Outliier Removal")
matplotlib.pyplot.show()

#Total Row is the outlier, others are valid records and will keep them for analysis

data_dict.pop( 'TOTAL', 0 )
data = featureFormat(data_dict, features)

#Plot after outlier removal
for point in data:
    salary = point[0]
    bonus = point[1]
    #print(salary)
    matplotlib.pyplot.scatter( salary, bonus )

matplotlib.pyplot.xlabel("salary")
matplotlib.pyplot.ylabel("bonus")
matplotlib.pyplot.title("After Outlier Removal")
matplotlib.pyplot.show()

### Task 3: Create new feature(s)
### Store to my_dataset for easy export below.
def computeFraction( poi_messages, all_messages ):
    """ given a number messages to/from POI (numerator) 
        and number of all messages to/from a person (denominator),
        return the fraction of messages to/from that person
        that are from/to a POI
   """


    ### you fill in this code, so that it returns either
    ###     the fraction of all messages to this person that come from POIs
    ###     or
    ###     the fraction of all messages from this person that are sent to POIs
    ### the same code can be used to compute either quantity

    ### beware of "NaN" when there is no known email address (and so
    ### no filled email features), and integer division!
    ### in case of poi_messages or all_messages having "NaN" value, return 0.
    fraction = 0.
    if poi_messages == 'Nan' or all_messages == 'NaN':
        pass
    else:
        fraction = float(poi_messages)/float(all_messages)


    return fraction


submit_dict = {}
for name in data_dict:

    data_point = data_dict[name]

    #print
    from_poi_to_this_person = data_point["from_poi_to_this_person"]
    to_messages = data_point["to_messages"]
    fraction_from_poi = computeFraction( from_poi_to_this_person, to_messages )
    #print fraction_from_poi
    data_point["fraction_from_poi"] = fraction_from_poi


    from_this_person_to_poi = data_point["from_this_person_to_poi"]
    from_messages = data_point["from_messages"]
    fraction_to_poi = computeFraction( from_this_person_to_poi, from_messages )
    #print fraction_to_poi
    submit_dict[name]={"from_poi_to_this_person":fraction_from_poi,
                       "from_this_person_to_poi":fraction_to_poi}
    data_point["fraction_to_poi"] = fraction_to_poi

for i in data_dict:
    data_dict[i]["fraction_from_poi"]=submit_dict[i]["from_poi_to_this_person"]
    data_dict[i]["fraction_to_poi"]=submit_dict[i]["from_this_person_to_poi"]
    
#features_list_old = features_list
features_list.append('fraction_from_poi')
features_list.append('fraction_to_poi')

    
### Store to my_dataset for easy export below.
my_dataset = data_dict

data = featureFormat(my_dataset, features_list, sort_keys = True)


for point in data:
    fraction_from_poi = point[20]
    fraction_to_poi = point[21]
    if point[0] == 1:
        matplotlib.pyplot.scatter( fraction_from_poi, fraction_to_poi, color = 'r' )
    else:
        matplotlib.pyplot.scatter( fraction_from_poi, fraction_to_poi, color = 'b' )
matplotlib.pyplot.xlabel("Fraction from POI")
matplotlib.pyplot.ylabel("Fraction to POI")
matplotlib.pyplot.title("Emails From POI Vs To POI")
matplotlib.pyplot.show()

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.
### Gaussian Naive Bayes
from sklearn.naive_bayes import GaussianNB
clf_NB = GaussianNB()  

### Adaboost Classifier
from sklearn.ensemble import AdaBoostClassifier
clf_AB = AdaBoostClassifier(algorithm= 'SAMME')

### Random Forest Classifier
from sklearn.ensemble import RandomForestClassifier
clf_RF = RandomForestClassifier(max_depth=2, random_state=0)

### Decision Tree Classifier
from sklearn.tree import DecisionTreeClassifier
clf_DT = DecisionTreeClassifier()



print 'tester_scale.py evaluator\n'
test_classifier(clf_NB,my_dataset,features_list)
test_classifier(clf_AB,my_dataset,features_list)
test_classifier(clf_RF,my_dataset,features_list)
test_classifier(clf_DT,my_dataset,features_list)


## Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html
# Feature importances of the DecisionTree Classifier
feature_importances = (clf_DT.feature_importances_)
features = zip(feature_importances, features_list[1:])
features = sorted(features, key= lambda x:x[0], reverse=True)

# Display the feature names and importance values
print('Tree Feature Importances:\n')
for i in range(10):
    print('{} : {:.4f}'.format(features[i][1], features[i][0]))
	
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)
import numpy as np
from sklearn.model_selection import GridSearchCV

no_of_features = np.arange(1, len(features_list))

# Create a pipeline with feature selection and classification
pipe = Pipeline([
    ('select_features', SelectKBest()),
    ('classify', DecisionTreeClassifier())
])

param_grid = [
    {
        'select_features__k': no_of_features
    }
]

# Use GridSearchCV to automate the process of finding the optimal number of features
dt_clf= GridSearchCV(pipe, param_grid=param_grid, scoring='f1', cv = 10)
dt_clf.fit(features, labels)

dt_clf.best_estimator_

def select_k_best(dictionary, features_list, k):
    
    data = featureFormat(dictionary, features_list)
    labels, features = targetFeatureSplit(data)

    k_best = SelectKBest(k=k)
    k_best.fit(features, labels)
    scores = k_best.scores_
    pairs = zip(features_list[1:], scores)
    #combined scores and features into a pandas dataframe then sort 
    k_best_features = pd.DataFrame(pairs,columns = ['feature','score'])
    k_best_features = k_best_features.sort_values('score',ascending = False)
    
    return k_best_features[:k]
    
k = 19 # used as per best_estimator from GridSearchCV
best_features = select_k_best(data_dict,features_list,k)

my_features_list = ['poi']
for i in best_features.feature:
    my_features_list.append(i)
    
# Create a pipeline with feature selection and classifier
tree_pipe = Pipeline([
    ('select_features', SelectKBest(k=19)),
    ('classify', DecisionTreeClassifier()),
])

# Define parameters to test with the 
# Decision Tree Classifier
param_grid = dict(classify__criterion = ['gini', 'entropy'] , 
                  classify__min_samples_split = [2, 4, 6, 8, 10, 20],
                  classify__max_depth = [None, 5, 10, 15, 20],
                  classify__max_features = [None, 'sqrt', 'log2', 'auto'])

# Use GridSearchCV to find the optimal parameters for the classifier
tree_clf = GridSearchCV(tree_pipe, param_grid = param_grid, scoring='recall', cv=10)
tree_clf.fit(features, labels)
# Get the best algorithm hyperparameters for the Decision Tree
tree_clf.best_params_

# Create the classifier with the optimal hyperparameters as found by GridSearchCV
data = featureFormat(my_dataset , my_features_list)
labels, features = targetFeatureSplit(data)

clf = Pipeline([
    ('select_features', SelectKBest(k=19)),
    ('classify', DecisionTreeClassifier(criterion='entropy', max_depth=15, max_features=None, min_samples_split=20))
])

test_classifier(clf,my_dataset,my_features_list)

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, my_features_list)