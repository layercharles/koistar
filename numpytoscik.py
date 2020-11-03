import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
koi = pd.read_csv('cumulative.csv')
datapc = ["koi_impact", "koi_prad", "koi_steff", "koi_depth", "koi_period", "koi_duration", "koi_slogg", "koi_srad", 'koi_teq']
koidf = pd.DataFrame(koi, columns=datapc + ["koi_disposition"])
koidf = koidf.replace(['CONFIRMED'],'1')
koidf = koidf.replace(['FALSE POSITIVE'],'2')
koidf = koidf.replace(['CANDIDATE'],'3')
X = koidf[datapc]
y = koidf['koi_disposition']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# X_train.fillna(X_train.mean())
# X_test.fillna(X_test.mean())
# y_train.fillna(y_train.mean())
# y_test.fillna(y_test.mean())
# print('Training Features Shape:', y_train.shape)

clf = RandomForestClassifier(n_estimators=1000)
X_train = np.nan_to_num(X_train)
y_train = np.nan_to_num(y_train)
X_test = np.nan_to_num(X_test)
y_test = np.nan_to_num(y_test)
clf.fit(X_train,y_train)
y_pred = clf.predict(X_test)

print("Accuracy:",metrics.accuracy_score(y_test, y_pred))


def plot_feature_importance(importance,names,model_type):

    #Create arrays from feature importance and feature names
    feature_importance = np.array(importance)
    feature_names = np.array(names)

    #Create a DataFrame using a Dictionary
    data={'feature_names':feature_names,'feature_importance':feature_importance}
    fi_df = pd.DataFrame(data)

    #Sort the DataFrame in order decreasing feature importance
    fi_df.sort_values(by=['feature_importance'], ascending=False,inplace=True)

    #Define size of bar plot
    plt.figure(figsize=(10,9))
    #Plot Searborn bar chart
    sns.barplot(x=fi_df['feature_importance'], y=fi_df['feature_names'])
    #Add chart labels
    plt.title(model_type + 'FEATURE IMPORTANCE')
    plt.xlabel('FEATURE IMPORTANCE')
    plt.ylabel('FEATURE NAMES')

plot_feature_importance(clf.feature_importances_,X.columns,'RANDOM FOREST')













# # koi.head(5)
# # print(koi.describe())
# koi = pd.get_dummies(koi)
# # koi.iloc[:,5:].head(5)
# labels = np.array(koi['koi_disposition_CONFIRMED'])
# koi = koi.drop('koi_disposition_CONFIRMED', axis = 1)

# koi_list = list(koi.columns)
# koi = np.array(koi)






# train_features, test_features, train_labels, test_labels = train_test_split(koi, labels, test_size = 0.1)

# print('Training Features Shape:', train_features.shape)
# print('Training Labels Shape:', train_labels.shape)
# print('Testing Features Shape:', test_features.shape)
# print('Testing Labels Shape:', test_labels.shape)

# # clf=RandomForestClassifier(n_estimators=100)
# # clf.fit(train_features, train_labels)

