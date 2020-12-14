import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
import random

#citire si parsare csv-uri
def read_csv(file):
	with open(file, 'r') as f:
		lines = f.readlines()

	lines = map(lambda line: line.strip('\n').split(','), lines)

	return lines

#functii folosite pentru a determina potrivirea pentru fiecare categorie
def eq(x, y):
	return x in y

def drink(x, y):
	if x == 'abstemious':
		return y[0] == 'No_Alcohol_Served'

	return y[0] != 'No_Alcohol_Served'

def dress_code(x, y):
	if x == 'no preference':
		return True

	return eq(x, y)

def smoking(x, y):
	d_true = {'none': False, 'only at bar': True, 'permitted': True,'section': True,'not permitted': False}
	d_false = {'none': False, 'only at bar': True, 'permitted': False,'section': True,'not permitted': True}
	
	if y[0] == '' or x == '':
		return False
	if x == 'true':
		return d_true[y[0]]
	return d_false[y[0]]


# determinarea potrivirii pentru fiecare categorie in parte pentru perechile (restaurant, client)
def get_matching(restaurants, users):
	#payment, cuisine, drink, budget, dress_code, smoking
	pairs = [[1, 2, eq], [2, 1, eq], [15, 6 ,drink], [19, 19, eq], [17, 7, dress_code], [16, 5, smoking]]

	matching = {}
	for rest in restaurants[1:]:
		for user in users[1:]:
			matching[(rest[0], user[0])] = []
			for pair in pairs:
				matching[(rest[0], user[0])].append(0)
				for l in user[pair[1]].split(';'):
					if pair[2](l, rest[pair[0]].split(';')):
						matching[(rest[0], user[0])][-1] += 1

	return matching


#citesc datele
restaurants = read_csv('restaurante.csv')
users = read_csv('utilizatori.csv')
ratings = read_csv('rating.csv')

#determina potrivirile intre restaunante si utilizatori
matching = get_matching(restaurants, users)

#construiesc datele pentru model
X = []
y = []

for r in ratings[1:]:
	if (r[1], r[0]) not in matching:
		print 'Rating invalid:' + (r[1], r[0])
	else:
		X.append(matching[(r[1], r[0])])
		y.append(int(r[2]))

#impart setul de date
X_y = zip(X, y)
random.shuffle(X_y)
X = map(lambda x:x[0], X_y)
y = map(lambda x:x[1], X_y)
X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False)

#antrenez modelul
model = LinearRegression().fit(X_train, y_train)

#determin rezultatele
print mean_squared_error(model.predict(X_train), y_train)
print mean_squared_error(model.predict(X_test), y_test)

count = count2 = 0
for (y1, y2) in zip(y_test, model.predict(X_test)):
	if y1 < 2 and int(round(y2)) > 1:
		count += 1
	if y1 == 2 and int(round(y2)) < 2:
		count2 += 1
print count, count2, len(y_test), float(count)/len(y_test), float(count2)/len(y_test)
