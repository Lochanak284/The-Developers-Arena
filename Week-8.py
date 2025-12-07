import os, sys, warnings, json, random
warnings.filterwarnings("ignore")
import numpy as np # type: ignore # type: ignore
import pandas as pd # type: ignore
import matplotlib.pyplot as plt # type: ignore
import seaborn as sns # type: ignore
from sklearn.model_selection import train_test_split, RandomizedSearchCV # type: ignore
from sklearn.ensemble import RandomForestClassifier # type: ignore
from sklearn.preprocessing import OneHotEncoder, StandardScaler # type: ignore
from sklearn.compose import ColumnTransformer # type: ignore
from sklearn.pipeline import Pipeline # type: ignore
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve, classification_report # type: ignore
from sklearn.calibration import CalibratedClassifierCV # type: ignore
from joblib import dump # type: ignore
from datetime import datetime

RND = 42
np.random.seed(RND)
random.seed(RND)

def generate_synthetic_telecom_data(n=5000, random_state=RND):
    np.random.seed(random_state)
    df = pd.DataFrame()
    df['customerID'] = [f"C{100000+i}" for i in range(n)]
    df['gender'] = np.random.choice(['Male','Female'], size=n)
    df['SeniorCitizen'] = np.random.choice([0,1], size=n, p=[0.88,0.12])
    df['Partner'] = np.random.choice(['Yes','No'], size=n, p=[0.45,0.55])
    df['Dependents'] = np.random.choice(['Yes','No'], size=n, p=[0.30,0.70])
    df['tenure_months'] = np.random.randint(0, 72, size=n)
    df['PhoneService'] = np.random.choice(['Yes','No'], size=n, p=[0.9,0.1])
    df['MultipleLines'] = np.random.choice(['Yes','No','No phone service'], size=n, p=[0.25,0.6,0.15])
    df['InternetService'] = np.random.choice(['DSL','Fiber optic','No'], size=n, p=[0.35,0.45,0.2])
    df['OnlineSecurity'] = np.random.choice(['Yes','No','No internet service'], size=n, p=[0.25,0.6,0.15])
    df['OnlineBackup'] = np.random.choice(['Yes','No','No internet service'], size=n, p=[0.25,0.6,0.15])
    df['DeviceProtection'] = np.random.choice(['Yes','No','No internet service'], size=n, p=[0.2,0.65,0.15])
    df['TechSupport'] = np.random.choice(['Yes','No','No internet service'], size=n, p=[0.2,0.65,0.15])
    df['StreamingTV'] = np.random.choice(['Yes','No','No internet service'], size=n, p=[0.25,0.6,0.15])
    df['StreamingMovies'] = np.random.choice(['Yes','No','No internet service'], size=n, p=[0.25,0.6,0.15])
    df['Contract'] = np.random.choice(['Month-to-month','One year','Two year'], size=n, p=[0.55,0.25,0.2])
    df['PaperlessBilling'] = np.random.choice(['Yes','No'], size=n, p=[0.6,0.4])
    df['PaymentMethod'] = np.random.choice(['Electronic check','Mailed check','Bank transfer (automatic)','Credit card (automatic)'], size=n)
    base = np.random.normal(25, 7, size=n)
    add_internet = (df['InternetService']=='Fiber optic')*30 + (df['InternetService']=='DSL')*10
    add_stream = (df['StreamingTV']=='Yes')*8 + (df['StreamingMovies']=='Yes')*8
    df['MonthlyCharges'] = np.clip(base + add_internet + add_stream + np.random.normal(0,5,n), 18, 200)
    df['TotalCharges'] = np.round(df['MonthlyCharges'] * (df['tenure_months'] + np.random.choice([0,1], p=[0.7,0.3], size=n)),2)
    churn_prob = (0.25*(df['Contract']=='Month-to-month').astype(int) + 0.15*(df['PaymentMethod']=='Electronic check').astype(int) + 0.002*(df['MonthlyCharges']) - 0.003*(df['tenure_months']))
    churn_prob = 1/(1+np.exp(- (churn_prob - 0.4)))
    df['Churn'] = np.where(np.random.rand(n) < churn_prob, 'Yes', 'No')
    for col in ['TotalCharges','MonthlyCharges']:
        mask = np.random.rand(n) < 0.01
        df.loc[mask, col] = np.nan
    return df

df = generate_synthetic_telecom_data(n=5000)

print("DATA SHAPE", df.shape)
print(df.head().to_string(index=False))
print("MISSING\n", df.isna().sum())
print("CHURN DISTRIBUTION\n", df['Churn'].value_counts(normalize=True))

TARGET = 'Churn'
y = df[TARGET].map({'Yes':1, 'No':0})
drop_cols = [c for c in df.columns if c.lower().startswith('customerid') or c.lower().endswith('id')]
X = df.drop(columns=drop_cols + [TARGET], errors='ignore')

def feature_engineering(df_in):
    df2 = df_in.copy()
    if 'MonthlyCharges' in df2.columns and 'tenure_months' in df2.columns:
        df2['charge_per_month_of_tenure'] = df2['MonthlyCharges'] / (df2['tenure_months'].replace(0,1))
    if 'InternetService' in df2.columns:
        df2['has_internet'] = (df2['InternetService'] != 'No').astype(int)
    return df2

numeric_cols = X.select_dtypes(include=['int64','float64']).columns.tolist()
categorical_cols = X.select_dtypes(include=['object','bool','category']).columns.tolist()

X_proc = X.copy()
for c in numeric_cols:
    med = X_proc[c].median()
    X_proc[c] = X_proc[c].fillna(med)
for c in categorical_cols:
    X_proc[c] = X_proc[c].fillna('Missing')

X_proc = feature_engineering(X_proc)

numeric_cols = X_proc.select_dtypes(include=['int64','float64']).columns.tolist()
categorical_cols = X_proc.select_dtypes(include=['object','bool','category']).columns.tolist()

def build_preprocessing_pipeline(numeric_cols, categorical_cols):
    categorical_transformer = Pipeline(steps=[('ohe', OneHotEncoder(sparse=False, handle_unknown='ignore'))])
    preprocessor = ColumnTransformer(transformers=[('num', 'passthrough', numeric_cols), ('cat', categorical_transformer, categorical_cols)], remainder='drop')
    return preprocessor

preprocessor = build_preprocessing_pipeline(numeric_cols, categorical_cols)

X_train, X_test, y_train, y_test = train_test_split(X_proc, y, stratify=y, test_size=0.2, random_state=RND)

clf = RandomForestClassifier(random_state=RND, n_jobs=-1)
pipe = Pipeline(steps=[('preprocessor', preprocessor), ('scaler', StandardScaler()), ('clf', clf)])

param_distributions = {'clf__n_estimators': [100, 200], 'clf__max_depth': [6, 10, None], 'clf__min_samples_split': [2,5,10], 'clf__class_weight': [None, 'balanced']}

rs = RandomizedSearchCV(pipe, param_distributions, n_iter=6, scoring='roc_auc', cv=3, random_state=RND, verbose=0)
rs.fit(X_train, y_train)
best_model = rs.best_estimator_
calibrated = CalibratedClassifierCV(base_estimator=best_model, cv='prefit')
calibrated.fit(X_train, y_train)
model = calibrated

def evaluate_model(model, X_t, y_t, prefix="Test"):
    y_pred = model.predict(X_t)
    y_proba = model.predict_proba(X_t)[:,1]
    acc = accuracy_score(y_t, y_pred)
    prec = precision_score(y_t, y_pred)
    rec = recall_score(y_t, y_pred)
    f1 = f1_score(y_t, y_pred)
    roc = roc_auc_score(y_t, y_proba)
    print(f"{prefix} metrics Accuracy:{acc:.3f} Precision:{prec:.3f} Recall:{rec:.3f} F1:{f1:.3f} ROC_AUC:{roc:.3f}")
    print(classification_report(y_t, y_pred, digits=3))
    cm = confusion_matrix(y_t, y_pred)
    print("Confusion matrix\n", cm)
    fpr, tpr, thresholds = roc_curve(y_t, y_proba)
    plt.figure(figsize=(6,4))
    plt.plot(fpr, tpr, label=f'ROC AUC={roc:.3f}')
    plt.plot([0,1],[0,1],'--')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"{prefix} ROC Curve")
    plt.legend()
    plt.savefig("roc_curve.png")
    plt.close()
    return {'accuracy':acc, 'precision':prec, 'recall':rec, 'f1':f1, 'roc_auc':roc, 'y_proba':y_proba}

metrics = evaluate_model(model, X_test, y_test, prefix="Test")

def get_feature_names_after_preprocessing(preprocessor, X_sample):
    preprocessor.fit(X_sample)
    names = []
    for nm in preprocessor.transformers_[0][2]:
        names.append(nm)
    cat_pipe = preprocessor.transformers_[1][1]
    ohe = cat_pipe.named_steps['ohe']
    cat_names = []
    if hasattr(ohe, 'get_feature_names_out'):
        cat_names = list(ohe.get_feature_names_out(preprocessor.transformers_[1][2]))
    else:
        for col in preprocessor.transformers_[1][2]:
            cat_names.append(col)
    names.extend(cat_names)
    return names

feature_names = get_feature_names_after_preprocessing(preprocessor, X_train)
try:
    importances = best_model.named_steps['clf'].feature_importances_
    if len(importances) == len(feature_names):
        feat_imp = pd.Series(importances, index=feature_names).sort_values(ascending=False).head(20)
        print("Top feature importances\n", feat_imp)
        plt.figure(figsize=(8,6))
        feat_imp.plot(kind='barh')
        plt.gca().invert_yaxis()
        plt.title("Top feature importances")
        plt.tight_layout()
        plt.savefig("feature_importances.png")
        plt.close()
except Exception:
    pass

cost_of_contact = 50.0
expected_gain_if_retained = 500.0
net_gain_if_true_positive = expected_gain_if_retained - cost_of_contact
net_loss_if_false_positive = -cost_of_contact
p_threshold = (-net_loss_if_false_positive) / (net_gain_if_true_positive - net_loss_if_false_positive)
print("Business probability threshold", round(p_threshold,3))

y_proba_test = metrics['y_proba']
y_pred_thresh = (y_proba_test >= p_threshold).astype(int)
print("Precision at threshold", precision_score(y_test, y_pred_thresh))
print("Recall at threshold", recall_score(y_test, y_pred_thresh))
print("F1 at threshold", f1_score(y_test, y_pred_thresh))
print("Confusion matrix at threshold\n", confusion_matrix(y_test, y_pred_thresh))

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
model_filename = f"telecom_churn_model_{timestamp}.joblib"
dump({'model': model, 'preprocessor': preprocessor, 'numeric_cols': numeric_cols, 'categorical_cols': categorical_cols}, model_filename)
report = {'timestamp': timestamp, 'dataset_shape': df.shape, 'train_shape': X_train.shape, 'test_shape': X_test.shape, 'churn_rate_train': float(y_train.mean()), 'churn_rate_test': float(y_test.mean()), 'metrics_test': {k: float(v) for k,v in metrics.items() if k in ['accuracy','precision','recall','f1','roc_auc']}, 'business_threshold': float(p_threshold)}
with open(f"project_summary_{timestamp}.json","w") as f:
    json.dump(report, f, indent=2)
print("Saved files:", "roc_curve.png", "feature_importances.png (if created)", model_filename, f"project_summary_{timestamp}.json")
