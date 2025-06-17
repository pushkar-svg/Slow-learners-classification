
import numpy as np

dataset_path = r"/content/dp dataset.csv"
data = pd.read_csv(dataset_path)

X = data.iloc[:, :-1]  
y = data.iloc[:, -1]   


for col in X.select_dtypes(include=np.number).columns:
    X[col].fillna(X[col].mean(), inplace=True)
for col in X.select_dtypes(include=['object']).columns:
    X[col].fillna(X[col].mode()[0], inplace=True)


numeric_features = X.select_dtypes(include=np.number).columns
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X[numeric_features])

X_scaled = pd.DataFrame(X_scaled, columns=numeric_features, index=X.index)

X_processed = pd.concat([X_scaled, X.select_dtypes(exclude=np.number)], axis=1)

X_train, X_test, y_train, y_test = train_test_split(X_processed, y, test_size=0.2, random_state=42)

svm_model = SVC(kernel='linear', random_state=42, probability=True)  
svm_model.fit(X_train[numeric_features], y_train) 


y_pred = svm_model.predict(X_test[numeric_features])  
print(classification_report(y_test, y_pred))


def model_predict(data):
    return svm_model.predict_proba(data)[:, 1] 

explainer = shap.KernelExplainer(model_predict, X_train[numeric_features])

shap_values = explainer.shap_values(X_test[numeric_features])

shap.initjs()  
shap.summary_plot(shap_values, X_test[numeric_features], feature_names=numeric_features) # Specify feature names
