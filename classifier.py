from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Create the dataset using make_moons()
X, y = make_moons(n_samples=10000, noise=0.4)

# Split the dataset into training and test subsets using the train_test_split() method
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Use a Decision Tree Classifier and investigate the operation of the tree for entropy and Gini coefficient and different depths of the tree
tree_entropy = DecisionTreeClassifier(criterion='entropy', max_depth=5, random_state=42)
tree_entropy.fit(X_train, y_train)

tree_gini = DecisionTreeClassifier(criterion='gini', max_depth=5, random_state=42)
tree_gini.fit(X_train, y_train)

# Use Random Forests as the classifier and test the performance of the classifier for a different number of decision trees
rf_10 = RandomForestClassifier(n_estimators=10, max_depth=5, random_state=42)
rf_10.fit(X_train, y_train)

rf_50 = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42)
rf_50.fit(X_train, y_train)

# Train Logistic Regression Classifier and SVM
log_reg = LogisticRegression(random_state=42)
log_reg.fit(X_train, y_train)

svm = SVC(kernel='linear', random_state=42)
svm.fit(X_train, y_train)

# Combine the SVM, Logistic Regression and Random Forest classifiers into one group (VotingClassifier) to improve the classification
voting_clf = VotingClassifier(estimators=[('lr', log_reg), ('rf', rf_50), ('svc', svm)], voting='hard')
voting_clf.fit(X_train, y_train)

# Evaluate the results achieved
print("Decision Tree - Entropy score: ", accuracy_score(y_test, tree_entropy.predict(X_test)))
print("Decision Tree - Gini score: ", accuracy_score(y_test, tree_gini.predict(X_test)))
print("Random Forest (10 trees) score: ", accuracy_score(y_test, rf_10.predict(X_test)))
print("Random Forest (50 trees) score: ", accuracy_score(y_test, rf_50.predict(X_test)))
print("Logistic Regression score: ", accuracy_score(y_test, log_reg.predict(X_test)))
print("SVM score: ", accuracy_score(y_test, svm.predict(X_test)))
print("Voting Classifier score: ", accuracy_score(y_test, voting_clf.predict(X_test)))
