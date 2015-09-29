#!/usr/bin/python

import sys
import pickle
sys.path.append("../tools/")
import matplotlib.pyplot
from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".

### All possible features ['poi','salary', 'deferral_payments', 'total_payments', 'loan_advances', 'bonus', 'restricted_stock_deferred', 'deferred_income', 'total_stock_value', 'expenses', 'exercised_stock_options', 'other', 'long_term_incentive', 'restricted_stock', 'director_fees', 'to_messages', 'email_address', 'from_poi_to_this_person', 'from_messages', 'from_this_person_to_poi', 'shared_receipt_with_poi']

features_list = ['poi', 'expenses', 'exercised_stock_options', 'other', 'from_this_person_to_poi']

### Load the dictionary containing the dataset
data_dict = pickle.load(open("final_project_dataset.pkl", "r") )

### Task 2: Remove outliers
data_dict.pop('TOTAL', 0)

### Ran some EDA to find what proportion of the 145 data points were POIs
#x = 0
#y= 'from_messages'
#for k, v in data_dict.iteritems():
#    if v['poi']:
#        x += 1
#print 18./145

### Used this to do some EDA and visualize some of the features
#data = featureFormat(data_dict, features_list)
#use_colors = {0 : 'red', 1 : 'blue'}
#for point in data:
#    salary = point[1]
#    bonus = point[2]
#    matplotlib.pyplot.scatter( salary, bonus, c=[use_colors[point[0]]])
#matplotlib.pyplot.xlabel("salary")
#matplotlib.pyplot.ylabel("bonus")
#matplotlib.pyplot.show()


### Task 3: Create new feature(s)
### Tried creating the following features, but none helped the model performance,
### so I commented them out
#for _, v in data_dict.iteritems():
#    try:
#        v['percent_stock_exercised'] = v['exercised_stock_options']*1.0/v['total_stock_value']
#    except TypeError:
#        v['percent_stock_exercised'] = 'NaN'
#    try:
#        v['percent_other_payments'] = v['other']*1.0/v['total_payments']
#    except TypeError:
#        v['percent_other_payments'] = 'NaN'
#    try:
#        v['bonus_sal_ratio'] = v['bonus']*1.0/v['salary']
#    except TypeError:
#        v['bonus_sal_ratio'] = 'NaN'
#    try:
#        v['stock_payment_ratio'] = v['total_stock_value']*1.0/v['total_payments']
#    except TypeError:
#        v['stock_payment_ratio'] = 'NaN'
#    try:
#        v['mois'] = ((v['from_this_person_to_poi'] +
#                     v['from_poi_to_this_person'] +
#                     v['shared_receipt_with_poi']) * 1.0 /
#                    (v['to_messages'] + v['from_messages']))
#    except TypeError:
#        v['mois'] = 'NaN'



### Store to my_dataset for easy export below.
my_dataset = data_dict

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.
from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier(min_samples_leaf = 1)

### Other classifiers that I tried
#from sklearn.naive_bayes import GaussianNB
#clf = GaussianNB()
#from sklearn.svm import SVC
#clf = SVC(C=.00001)
#clf.fit(features, labels)
#from sklearn.ensemble import AdaBoostClassifier

### Code for feature selection and parameter tuning

#from sklearn.feature_selection import SelectKBest
#selection = SelectKBest(k=4)
#selection.fit(features, labels)
#print selection.get_support()
#from sklearn.grid_search import GridSearchCV
#dtc = DecisionTreeClassifier(random_state = 10)
#params = {'min_samples_leaf': [1, 2, 5, 8, 10, 15], 'max_depth': [2, 5, 10]}
#from sklearn.metrics import fbeta_score, make_scorer
#ftwo_scorer = make_scorer(fbeta_score, beta=1)
#clf = GridSearchCV(dtc, params, scoring='f1')
#clf.fit(features, labels)
#print clf.best_params_

### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

### I moved evaluation to testing.py script to get better stats and utilize cross-validation

# Example starting point. Try investigating other evaluation techniques!
#from sklearn.cross_validation import train_test_split
#features_train, features_test, labels_train, labels_test = \
#    train_test_split(features, labels, test_size = 0.3, random_state = 42)
#
#clf.fit(features_train, labels_train)
#pred = clf.predict(features_test)

#from sklearn.metrics import accuracy_score, roc_auc_score, average_precision_score, precision_score, recall_score
#auc = roc_auc_score(labels_test, pred)
#ap = average_precision_score(labels_test, pred)
#acc = accuracy_score(labels_test, pred)
#precision = precision_score(labels_test, pred)
#recall = recall_score(labels_test, pred)
#print acc, precision, recall
#print clf.feature_importances_

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)