##############################
# 1. Importing Dependencies  #
##############################

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler  
from sklearn.model_selection import StratifiedKFold, GridSearchCV, train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline as ImbPipeline


#######################################
# 2. Data Import and Initial Cleaning #
#######################################

df = pd.read_csv("../data/dataset.csv")
df_output = pd.read_csv("../data/outcome.csv")

# 2.1 Shape
print(f' The Dataset has {df.shape[0]} rows and {df.shape[1]} columns.') 

# 2.2 Data types
print(f'Data types: {df.dtypes}')

# 2.3 Encoding categorical features
categorical_cols = df.select_dtypes(include=['object']).columns
df_cat_encoded = pd.get_dummies(df[categorical_cols], drop_first=True).sort_index()
df_num = df.select_dtypes(include=[np.number]).sort_index()
df_final = pd.concat([df_num, df_cat_encoded], axis=1)

# 2.4 Preview
print(f' First 5 Rows: {df.head()}') 

# 2.5 NaN and duplicates
print(f' Amount of NaNs: {df.isnull().sum().sum()}') 
print(f' Count of duplicated datapoints: {df.duplicated().sum()}')
print(f' Count of duplicated features: {df.T.duplicated().sum()}')

# 2.6 Class distribution
print(f' Classes of the target variable: {df_output["BRCA_subtype"].value_counts()}')

# 2.7 Merge & cleanup
common_index = df_final.index.intersection(df_output.index)
df_final = df_final.loc[common_index].sort_index()
df_output = df_output.loc[common_index].sort_index()
df_cleaned_1 = df_final.loc[:, ~df_final.columns.duplicated()]


###########################################
# 3. Label Mapping and Class Balancing    #
###########################################

valid_classes = ['LumA', 'LumB', 'Her2', 'Basal', 'Normal']
df_valid = df_output[df_output['BRCA_subtype'].isin(valid_classes)].copy()

X = df_cleaned_1.loc[df_valid.index]
y = df_valid['BRCA_subtype'].map({'LumA': 0, 'LumB': 1, 'Her2': 2, 'Basal': 3, 'Normal': 4})

# 3.1 Undersampling & SMOTE
under_sampler = RandomUnderSampler(sampling_strategy={0: 70, 1: 70, 3: 70, 4: 70}, random_state=42)
X_under, y_under = under_sampler.fit_resample(X, y)

smote = SMOTE(sampling_strategy={2: 70}, k_neighbors=5, random_state=42)
X_hybrid, y_hybrid = smote.fit_resample(X_under, y_under)

print(" Neue Verteilung nach Hybrid-Sampling:")
print(pd.Series(y_hybrid).value_counts())


###################################
# 4. Exploratory Data Analysis    #
###################################

# 4.1 Target distribution plot
sns.countplot(x='BRCA_subtype', data=df_output)
plt.title('Distribution of BRCA subtypes')
plt.xlabel('BRCA subtypes')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('../output/BRCA_subtype_distribution.png')


##################################################
# 5. Feature Selection & Model Pipeline Setup    #
##################################################

pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('pca', PCA()),
    ('clf', RandomForestClassifier(class_weight={0: 1, 1: 3, 2: 2, 3: 1, 4: 1}, random_state=42))
])

param_grid = {
    'pca__n_components': [30, 50, 100],
    'clf__n_estimators': [100, 350],
    'clf__max_depth': [10, 20, None]
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
metrics = ["accuracy", "precision_macro", "recall_macro", "f1_macro", "roc_auc_ovr", "average_precision"]
results = {}

# 5.1 GridSearch
for metric in metrics:
    print(f"\n Running GridSearch for: {metric}")
    grid = GridSearchCV(pipe, param_grid, cv=cv, scoring=metric, n_jobs=-1)
    grid.fit(X_hybrid, y_hybrid)
    
    print(f" Finished: {metric}")
    print(f"Best Score for {metric}: {grid.best_score_:.4f}")
    print(f"Best Params for {metric}: {grid.best_params_}")
    
    results[metric] = {
        'score': grid.best_score_,
        'params': grid.best_params_,
        'model': grid.best_estimator_
    }


#####################################################
# 6. Cross-Validation Metric Varianzberechnung      #
#####################################################

cv_scores_summary = {}

print("\n Cross-Validation-Ergebnisse mit ± Standardabweichung:")
for metric in metrics:
    print(f"\n Metrik: {metric}")
    
    model = results[metric]['model']
    scores = cross_val_score(model, X_hybrid, y_hybrid, cv=cv, scoring=metric, n_jobs=-1)
    
    mean = np.mean(scores)
    std = np.std(scores)
    print(f"{metric:<20} | Mittelwert: {mean:.4f} | ± {std:.4f}")
    
    results[metric]['mean_cv_score'] = mean
    results[metric]['std_cv_score'] = std
    cv_scores_summary[metric] = {
        'Mean Score': mean,
        'Std Deviation': std
    }

df_metrics = pd.DataFrame(cv_scores_summary).T
print("\n Übersicht der Metriken mit Varianz:")
print(df_metrics)


#######################################
# 7. Testset-Evaluation & ConfMatrix  #
#######################################

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

clf = RandomForestClassifier()
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

labels = [0, 1, 2, 3, 4]
cm_normal = confusion_matrix(y_test, y_pred, labels=labels)
disp_normal = ConfusionMatrixDisplay(confusion_matrix=cm_normal, display_labels=labels)
disp_normal.plot(cmap="Blues", xticks_rotation=45)
plt.title("Confusion Matrix RF")
plt.tight_layout()
plt.savefig("../output/confusion_matrix_RF.png")
plt.close()


##############################################
# 8. Evaluation pro Modell und Metrik        #
##############################################

print("\n Klassenspezifische Auswertung aller Modelle:")
for metric in metrics:
    print(f"\n Metrik: {metric}")
    best_model = results[metric]['model']
    best_model.fit(X_train, y_train)
    y_pred = best_model.predict(X_test)
    target_names = ['LumA', 'LumB', 'Her2', 'Basal', 'Normal']
    print(classification_report(y_test, y_pred, target_names=target_names, labels=labels))
    
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=target_names)
    disp.plot(cmap="Blues", xticks_rotation=45)
    plt.title(f"Confusion Matrix RF")
    plt.tight_layout()
    plt.savefig(f"../output/confusion_matrix_{metric}.png")
    plt.close()


#########################################
# 9. ROC AUC Kurven (Multiclass, CV)    #
#########################################

from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import cross_val_predict

classes = [0, 1, 2, 3, 4]
labels = ['LumA', 'LumB', 'Her2', 'Basal', 'Normal']
n_classes = len(classes)

y_bin = label_binarize(y, classes=classes)

ovr_clf = OneVsRestClassifier(
    RandomForestClassifier(class_weight={0: 1, 1: 3, 2: 2, 3: 1, 4: 1}, random_state=42)
)

y_score = cross_val_predict(ovr_clf, X, y, cv=cv, method='predict_proba', n_jobs=-1)

fpr, tpr, roc_auc = {}, {}, {}
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_bin[:, i], y_score[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

plt.figure(figsize=(8, 6))
for i in range(n_classes):
    plt.plot(fpr[i], tpr[i], lw=2, label=f"{labels[i]} (AUC = {roc_auc[i]:.2f})")

plt.plot([0, 1], [0, 1], 'k--', lw=1)
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Multiclass ROC-AUC Curve (Cross-Validated)")
plt.legend(loc="lower right")
plt.grid(True)
plt.tight_layout()
plt.savefig("../output/roc_auc_cv_multiclass.png")
plt.close()
