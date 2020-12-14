import numpy as np
import pandas as pan
from time import time

from sklearn.cluster import Birch
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder


def splitColumns(csvRead, columns):
    for column in columns:
        for rowCounter in range(0, int(csvRead.size/csvRead.columns.size)):
            while len(csvRead[column][rowCounter]) != 0:
                newColumn = csvRead[column][rowCounter][0]
                if newColumn not in csvRead.columns:
                    csvRead[newColumn] = np.zeros((int(csvRead.size/csvRead.columns.size),), dtype=int)
                csvRead.loc[rowCounter, newColumn] = 1
                csvRead[column][rowCounter].pop(0)


# read CSV file
usersFromCSV = pan.read_csv("data/utilizatori.csv")[['Ucuisine', 'Upayment', 'smoker', 'drink_level', 'dress_preference', 'budget']]
#usersFromCSV = pan.read_csv("data/utilizatori.csv")[['userID', 'Ucuisine', 'Upayment', 'smoker', 'drink_level', 'dress_preference', 'budget']]
#usersFromCSV = pan.read_csv("data/utilizatori.csv")[['userID', 'Ucuisine']]
print(usersFromCSV)


# convert each raw cuisine to array of cuisines
for counter in range(0, usersFromCSV['Ucuisine'].size):
    usersFromCSV['Ucuisine'][counter] = usersFromCSV['Ucuisine'][counter].split(';')
    usersFromCSV['Upayment'][counter] = usersFromCSV['Upayment'][counter].split(';')

# manually apply One Hot Encoding
splitColumns(usersFromCSV, ['Ucuisine', 'Upayment'])

# remove the original columns
del usersFromCSV['Ucuisine']
del usersFromCSV['Upayment']

le = LabelEncoder()

#usersFromCSV['userID'] = le.fit_transform(usersFromCSV['userID'].values)
usersFromCSV['smoker'] = le.fit_transform(usersFromCSV['smoker'].values)
usersFromCSV['drink_level'] = le.fit_transform(usersFromCSV['drink_level'].values)
usersFromCSV['dress_preference'] = le.fit_transform(usersFromCSV['dress_preference'].values)
usersFromCSV['budget'] = le.fit_transform(usersFromCSV['budget'].values)
print(usersFromCSV)

# threshold value keeps changing in order to find a proper value for Birch
thresholdValue = 0.1
thresholdValueFound = False
# Compute clustering with Birch with and without the final clustering step
while True:
    thresholdValue += 0.01
    birch_models = [Birch(threshold=thresholdValue, n_clusters=None),
                    Birch(threshold=thresholdValue, n_clusters=30)]
    final_step = ['without global clustering', 'with global clustering']

    for ind, (birch_model, info) in enumerate(zip(birch_models, final_step)):
        t = time()
        birch_model.fit(usersFromCSV)
        time_ = time() - t
        print("Birch %s as the final step took %0.2f seconds" % (
              info, (time() - t)))

        labels = birch_model.labels_
        centroids = birch_model.subcluster_centers_
        n_clusters = np.unique(labels).size
        print("n_clusters : %d" % n_clusters)

        ### Metric scores
        silhouetteScore = metrics.silhouette_score(usersFromCSV, labels, metric='euclidean', sample_size=(int(usersFromCSV.size/usersFromCSV.columns.size)))
        daviesBouldinScore = metrics.davies_bouldin_score(usersFromCSV, labels)
        calinskiHarabaszScore = metrics.calinski_harabasz_score(usersFromCSV, labels)

        print("Silhouette score: %.3f" % silhouetteScore)
        print("Davies Bouldin score: %.3f" % daviesBouldinScore)
        print("Calinski Harabasz score: %.3f" % calinskiHarabaszScore)

        print("Threshold: %.2f", thresholdValue)
        # good score check => end of the loop
        if (silhouetteScore > 0.7 and daviesBouldinScore < 0.2) and (daviesBouldinScore != 0 and silhouetteScore != 0):
            thresholdValueFound = True
    if thresholdValueFound:
        break




