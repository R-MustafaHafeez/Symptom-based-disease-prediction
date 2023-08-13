### import packages

## lets basic import packages

import numpy as np #libraries used for arrays 
import pandas as pd # mostly for data reading from csv files
import matplotlib.pyplot as plt # used for Data visualization like ploting graphs etc
import seaborn as sns #for making statistical graphics and easy to integrate with panda 
from sklearn.model_selection import train_test_split #split data into train and test 
from sklearn.utils import shuffle #used for shuffling attributes
from sklearn import metrics 
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score





## import data

df = pd.read_csv('dataset.csv')
df = shuffle(df, random_state = 42)
df.head()




## remove ('_') underscore in the text

for col in df.columns:
    df[col] = df[col].str.replace('_', ' ')
    
df.head()




## charactieristics of data

df.describe()



## check null values

null_checker = df.apply(lambda x: sum(x.isnull())).to_frame(name='count')
print(null_checker)




## plot of null value

plt.figure(figsize=(5, 3), dpi=140)
plt.plot(null_checker.index, null_checker['count'])
plt.xticks(null_checker.index, null_checker.index, rotation = 45, horizontalalignment = 'right')
plt.title('Ratio of Null values')
plt.xlabel('column names')
plt.margins(0.1)
plt.show()




cols = df.columns

data = df[cols].values.flatten()

reshaped = pd.Series(data)
reshaped = reshaped.str.strip()
reshaped = reshaped.values.reshape(df.shape)

df = pd.DataFrame(reshaped, columns = df.columns)
df.head()




## lets fill nan values

df = df.fillna(0)
df.head()




## lets explore symptom severity

df_severity = pd.read_csv('Symptom-severity.csv')
df_severity['Symptom'] = df_severity['Symptom'].str.replace('_',' ')
df_severity.head(10)




## overall list

df_severity['Symptom'].unique()



## lets encode sysptoms in the data

vals = df.values
symptoms = df_severity['Symptom'].unique()

for i in range(len(symptoms)):
    vals[vals == symptoms[i]] = df_severity[df_severity['Symptom'] == symptoms[i]]['weight'].values[0]




df_processed = pd.DataFrame(vals, columns=cols)
df_processed.head()




## assign symptoms with no rank to zero

df_processed = df_processed.replace('dischromic  patches', 0)
df_processed = df_processed.replace('spotting  urination', 0)
df_processed = df_processed.replace('foul smell of urine', 0)


## split data

data = df_processed.iloc[:,1:].values
labels = df['Disease'].values




## split train and test data

# help(train_test_split)

X_train, X_test, y_train, y_test = train_test_split(data, 
                                                    labels, 
                                                    test_size=0.2, 
                                                    random_state=42)



print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)


print(X_train[0])
print(X_test[0])
print(y_train[0])
print (y_test[0])




#implementation of decision tree 
from sklearn.tree import DecisionTreeClassifier
tree = DecisionTreeClassifier()

print (tree.fit(X_train, y_train))

print (tree.score(X_train, y_train))
tree.predict([[3,5,3,5,4,4,3,2,3,0,0,0,0,0,0,0,0]])



#finding Accuracy
from sklearn.tree import DecisionTreeClassifier
tree1 = DecisionTreeClassifier(criterion="entropy")
#training model 
tree1.fit(X_train,y_train)
print("Accuracy on training set: {:.3f}".format(tree1.score(X_train, y_train)))
print("Accuracy on test set: {:.3f}".format(tree1.score(X_test, y_test)))
#prediction on model
prediction = tree1.predict(X_test)
print("Predicted values:", prediction)



from sklearn.tree import plot_tree
plt.figure(figsize=(8,8))
plot_tree(tree1)



