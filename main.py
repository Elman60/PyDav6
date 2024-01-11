import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
from sklearn.impute import SimpleImputer

# ჩავტვირთეთ აშშ-ს უძრავი ქონების მონაცემები CSV ფაილიდან
file_path = 'realtor-data.csv'
df = pd.read_csv(file_path)

# ცარიელ ველებს ვანაცვლებთ საშუალოთი, კოდმა რომ იმუშავოს
imputer = SimpleImputer(strategy='mean')
df[['bed', 'bath', 'acre_lot', 'house_size']] = imputer.fit_transform(df[['bed', 'bath', 'acre_lot', 'house_size']])

y_imputer = SimpleImputer(strategy='mean')
df['price'] = y_imputer.fit_transform(df[['price']])

# მონაცემებს ვყოფთ (X) და სამიზნე (y) ცვლადებად
X = df[['bed', 'bath', 'acre_lot', 'house_size']]
y = df['price']

# ვყოფთ მონაცემებს სასწავლო და სატესტოდ
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 1. ერთ ცვლადიანი რეგრესიის მოდელი იქმნება და მას ვტესტავთ "house_size" ფუნქციის გამოყენებით. prediction კეთდება ორივე კომპლექტზე.

simple_linear_model = LinearRegression()
simple_linear_model.fit(X_train[['house_size']], y_train)
simple_linear_predictions_train = simple_linear_model.predict(X_train[['house_size']])
simple_linear_predictions_test = simple_linear_model.predict(X_test[['house_size']])
simple_linear_efficiency_train = mean_squared_error(y_train, simple_linear_predictions_train)
simple_linear_efficiency_test = mean_squared_error(y_test, simple_linear_predictions_test)

print("Simple Linear Regression Training Efficiency:", simple_linear_efficiency_train)
print("Simple Linear Regression Testing Efficiency:", simple_linear_efficiency_test)

# 2. მრავალცვლადიანი რეგრესიის მოდელი
multiple_linear_model = LinearRegression()
multiple_linear_model.fit(X_train, y_train)
multiple_linear_predictions_train = multiple_linear_model.predict(X_train)
multiple_linear_predictions_test = multiple_linear_model.predict(X_test)
multiple_linear_efficiency_train = mean_squared_error(y_train, multiple_linear_predictions_train)
multiple_linear_efficiency_test = mean_squared_error(y_test, multiple_linear_predictions_test)

print("Multiple Linear Regression Training Efficiency:", multiple_linear_efficiency_train)
print("Multiple Linear Regression Testing Efficiency:", multiple_linear_efficiency_test)

# 3. fit მეთოდის გამოყენებით გადაწყვეტილების ხის მოდელს ვამზადებთ მოცემულ მონაცემებზე.
decision_tree_model = DecisionTreeRegressor()
decision_tree_model.fit(X_train[['house_size']], y_train)
decision_tree_predictions_train = decision_tree_model.predict(X_train[['house_size']])
decision_tree_predictions_test = decision_tree_model.predict(X_test[['house_size']])
decision_tree_efficiency_train = mean_squared_error(y_train, decision_tree_predictions_train)
decision_tree_efficiency_test = mean_squared_error(y_test, decision_tree_predictions_test)

print("Decision Tree Regression Training Efficiency:", decision_tree_efficiency_train)
print("Decision Tree Regression Testing Efficiency:", decision_tree_efficiency_test)

# სახლის ფასების პროგნოზირება ახალი მონაცემებისთვის
new_data = pd.DataFrame({'house_size': [2000, 2500, 3000],
                         'bed': [3, 4, 2],
                         'bath': [2, 3, 2],
                         'acre_lot': [0.25, 0.3, 0.2]})

# გამოტოვებული მნიშვნელობებს ვამუშავებთ new_data-ში
new_data[['bed', 'bath', 'acre_lot', 'house_size']] = imputer.transform(new_data[['bed', 'bath', 'acre_lot', 'house_size']])


new_data = new_data[['bed', 'bath', 'acre_lot', 'house_size']]

simple_linear_predictions_new = simple_linear_model.predict(new_data[['house_size']])
multiple_linear_predictions_new = multiple_linear_model.predict(new_data)
decision_tree_predictions_new = decision_tree_model.predict(new_data[['house_size']])

print("\nPredictions for new data:")
print("Simple Linear Regression:", simple_linear_predictions_new)
print("Multiple Linear Regression:", multiple_linear_predictions_new)
print("Decision Tree Regression:", decision_tree_predictions_new)
