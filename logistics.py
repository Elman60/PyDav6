import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder

# ჩავტვირთეთ საბანკო სესხის მონაცემები CSV ფაილიდან
file_path = 'LoanStatus.csv'
df = pd.read_csv(file_path)

#პირველი რამდენიმე სტრიქონის გამოტანა
print(df.head())

#გადავიყვანეთ კატეგორიული ცვლადები რიცხვითში
label_encoder = LabelEncoder()
categorical_columns = ['Gender', 'Married', 'Education', 'Loan_Status']
for column in categorical_columns:
    df[column] = label_encoder.fit_transform(df[column])

# გამოტოვებული მნიშვნელობებს ვამუშავებთ
imputer = SimpleImputer(strategy='mean')
df[['ApplicantIncome', 'LoanAmount', 'Loan_Amount_Term']] = imputer.fit_transform(df[['ApplicantIncome', 'LoanAmount', 'Loan_Amount_Term']])

# ვყოფთ დატასეტს X da Y-ად
X = df[['Gender', 'Married', 'Education', 'ApplicantIncome', 'LoanAmount', 'Loan_Amount_Term']]
y = df['Loan_Status']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. ლოგისტიკური რეგრესია
logistic_model = LogisticRegression()
logistic_model.fit(X_train, y_train)
logistic_predictions_test = logistic_model.predict(X_test)
logistic_efficiency_test = accuracy_score(y_test, logistic_predictions_test)

print("Logistic Regression Testing Efficiency:", logistic_efficiency_test)
print("Classification Report (Testing):\n", classification_report(y_test, logistic_predictions_test))

# ახალი მონაცემის შეყვანა
new_data = pd.DataFrame({'Gender': [1, 0, 1],  # 1 for Male, 0 for Female
                         'Married': [1, 0, 1],  # 1 for Yes, 0 for No
                         'Education': [1, 0, 1],  # 1 for Graduate, 0 for Not Graduate
                         'ApplicantIncome': [5000, 6000, 7000],
                         'LoanAmount': [200, 250, 300],
                         'Loan_Amount_Term': [360, 360, 360]})

new_data[['ApplicantIncome', 'LoanAmount', 'Loan_Amount_Term']] = imputer.transform(new_data[['ApplicantIncome', 'LoanAmount', 'Loan_Amount_Term']])

# პროგნოზირება ახალი მონაცემებით
logistic_predictions_new = logistic_model.predict(new_data)
print("\nLogistic Regression Predictions for new data:")
print(logistic_predictions_new)

# 5. გადაწყვეტილების ხის კლასიფიკაციის მოდელი ერთ ცვლადზე
decision_tree_classifier_model_single = DecisionTreeClassifier()
decision_tree_classifier_model_single.fit(X_train[['ApplicantIncome']], y_train)
decision_tree_classifier_predictions_test_single = decision_tree_classifier_model_single.predict(X_test[['ApplicantIncome']])
decision_tree_classifier_efficiency_test_single = accuracy_score(y_test, decision_tree_classifier_predictions_test_single)

print("\nDecision Tree Classification Efficiency (Income - Testing):", decision_tree_classifier_efficiency_test_single)
print("Classification Report (Income - Testing):\n", classification_report(y_test, decision_tree_classifier_predictions_test_single))

decision_tree_classifier_predictions_new_single = decision_tree_classifier_model_single.predict(new_data[['ApplicantIncome']])
print("\nDecision Tree Classification Predictions for new data (Income):")
print(decision_tree_classifier_predictions_new_single)



