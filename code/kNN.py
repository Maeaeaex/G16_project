import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler  
from sklearn.model_selection import StratifiedKFold, GridSearchCV, train_test_split
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
import os
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_predict
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc  

base_dir = os.path.dirname(os.path.abspath(__file__))

######################## Data import ########################

df = pd.read_csv(f'{base_dir}/../data/dataset.csv')
df_output = pd.read_csv(f'{base_dir}/../data/outcome.csv')


########### initial overview and data preprocessing ###########

# 1.1 Shape of the dataset
print(f' The Dataset has {df.shape[0]} rows and {df.shape[1]} columns.') 

# 1.2 data types
print(f'Data types: {df.dtypes}')

## categorial encoding 
categorical_cols = df.select_dtypes(include=['object']).columns
df_cat_encoded = pd.get_dummies(df[categorical_cols], drop_first=True)  # One-hot encoding for categorical variables
df_cat_encoded = df_cat_encoded.sort_index()

df_num = df.select_dtypes(include=[np.number])
df_num = df_num.sort_index()

df_final = pd.concat([df_num, df_cat_encoded], axis=1) # combine the categorial and the numerical features 


# 1.3 First 5 rows
print(f' First 5 Rows: {df.head()}') 

# 1.4 Amount of NaNs
print(f' Amount of NaNs: {df.isnull().sum().sum()}') 

# 1.5 Count of dublicated datapoints:
print(f' Count of dublicated datapoints: {df.duplicated().sum()}')
print(f' Count of dublicated features: {df.T.duplicated().sum()}')

## Comment: There are dublicates in columns -> needs to be cleaned

# 1.6 Classes of the target variable
print(f' Classes of the target variable: {df_output["BRCA_subtype"].value_counts()}')

## Comment: Not all classes are equally represented. -> correction needed

# - Matching feature matrix and target vector by index
common_index = df_final.index.intersection(df_output.index) # drops redundant indices if missing either in df or df_output
df_final = df_final.loc[common_index].sort_index()
df_output = df_output.loc[common_index].sort_index()
df_cleaned_1 = df_final.loc[:, ~df_final.columns.duplicated()]


# - Convert target variable to numerical values
# Mapping und Filtern auf bekannte Klassen
valid_classes = ['LumA', 'LumB', 'Her2', 'Basal', 'Normal']
df_valid = df_output[df_output['BRCA_subtype'].isin(valid_classes)].copy() # stores only subclasses of the column "BRCA_subtype" which are part of the list given (valid_classes)

# Features und Ziel synchronisieren
X = df_cleaned_1.loc[df_valid.index]
y = df_valid['BRCA_subtype'].map({
    'LumA': 0,
    'LumB': 1,
    'Her2': 2,
    'Basal': 3, 
    'Normal': 4
})


# Ziel: alle Klassen auf 60 bringen
target_counts = {
    0: 100,  # LumA (unter)
    1: 100,  # LumB (unter)
    2: 100,  # Her2 (über)
    3: 100,  # Basal (unter)
    4: 100   # Normal (unter)
}

# 1. Unter-Sampling (LumA → 100)
under_sampler = RandomUnderSampler(sampling_strategy={0: 70, 1: 70, 3: 70, 4: 70}, random_state=42)
X_under, y_under = under_sampler.fit_resample(X, y)

# 2. SMOTE (alle anderen auf 100)

smote = SMOTE(sampling_strategy={2: 70}, k_neighbors=5, random_state=42)
X_hybrid, y_hybrid = smote.fit_resample(X_under, y_under)

# Kontrollausgabe
print("📊 Neue Verteilung nach Hybrid-Sampling:")
print(pd.Series(y_hybrid).value_counts())




#################### Exploratory Data Analysis (EDA) ####################

# 2.1 Visualizing the target class distribution
sns.countplot(x='BRCA_subtype', data=df_output)
plt.title('Distribution of BRCA subtypes')
plt.xlabel('BRCA subtypes')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('../output/BRCA_subtype_distribution.png')

## Comment: Not all classes are equally represented. -> correction needed when training the model




#################### Feature Selection / Dimensionality Reduction ####################

# Pipeline: Scaling → PCA → RF


# #pipe = Pipeline([
#     ('scaler', StandardScaler()),           # wichtig für PCA
#     ('pca', PCA()),                         # n_components wird getuned
#     ('clf', RandomForestClassifier(class_weight={0: 1, 1: 3, 2: 2, 3: 1, 4: 1}, random_state=42))
#])

# Grid: Tune PCA + RandomForest
# #param_grid = {
#     'pca__n_components': [30, 50, 100],
#     'clf__n_estimators': [100, 350],
#     'clf__max_depth': [10, 20, None]
# #}

# Stratified CV
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# GridSearch mit CV für mehrere Metriken
metrics = ["accuracy", "precision_macro", "recall_macro", "f1_macro", "roc_auc_ovr", "average_precision"]
results = {}

# #for metric in metrics:
#     print(f"\n🔍 Running GridSearch for: {metric}")
#     grid = GridSearchCV(pipe, param_grid, cv=cv, scoring=metric, n_jobs=-1)
#     grid.fit(X_hybrid, y_hybrid)
    
#     print(f"✅ Finished: {metric}")
#     print(f"Best Score for {metric}: {grid.best_score_:.4f}")
#     print(f"Best Params for {metric}: {grid.best_params_}")
    
#     results[metric] = {
#         'score': grid.best_score_,
#         'params': grid.best_params_,
#         'model': grid.best_estimator_
#     }

## Classification report
labels = ['LumA', 'LumB', 'Her2', 'Basal', 'Normal']  # anpassen, falls nötig
all_labels = [0, 1, 2, 3, 4]

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)



#################### kNN-Classification ####################

# 1. kNN-Pipeline: Scaling → PCA → kNN
pipe_knn = Pipeline([
    ('scaler', StandardScaler()),           # wichtig vor PCA und kNN
    ('pca', PCA()),                         # Anzahl der Komponenten wird getuned
    ('clf', KNeighborsClassifier())         # kNN-Klassifikator
])

# 2. Parameter-Grid für kNN
param_grid_knn = {
    'pca__n_components': [20, 25, 30],     # exakt wie bei RandomForest
    'clf__n_neighbors': [13, 15, 17, 19],          # typische Nachbarn-Zahlen
    'clf__weights': ['uniform', 'distance'], # Gewichtung der Nachbarn
}

# 3. GridSearchCV auf Hybrid-Daten für dieselben Metriken
results_knn = {}


for metric in metrics:
    print(f"\n🔍 Running kNN GridSearch for: {metric}")
    grid_knn = GridSearchCV(
        pipe_knn,
        param_grid_knn,
        cv=cv,
        scoring=metric,
        n_jobs=-1,
        return_train_score=False
    )
    grid_knn.fit(X_hybrid, y_hybrid)

    # Mittelwert und Standardabweichung des besten Scores
    best_mean = grid_knn.best_score_
    best_std  = grid_knn.cv_results_['std_test_score'][grid_knn.best_index_]
    print(f"✅ Finished kNN: {metric}")
    print(f"Best kNN Score for {metric}: {best_mean:.4f} ± {best_std:.4f}")

    best_params = grid_knn.best_params_
    print(f"Best kNN Params for {metric}: {best_params}")

    results_knn[metric] = {
        'score_mean': best_mean,
        'score_std':  best_std,
        'params':     best_params,
        'model':      grid_knn.best_estimator_
    }

# 4. Klassenspezifische Auswertung auf Testdaten
print("\n📋 kNN Klassenspezifische Auswertung:")
for metric in metrics:
    print(f"\n📌 Metrik: {metric}")
    best_knn = results_knn[metric]['model']
    best_knn.fit(X_train, y_train)
    y_pred_knn = best_knn.predict(X_test)

    # Classification Report
    print(classification_report(y_test, y_pred_knn, target_names=labels, labels=all_labels))

    # Confusion Matrix
    cm_knn = confusion_matrix(y_test, y_pred_knn, labels=all_labels)
    disp_knn = ConfusionMatrixDisplay(confusion_matrix=cm_knn, display_labels=labels)
    disp_knn.plot(cmap="Blues", xticks_rotation=45)
    plt.title(f"Confusion Matrix kNN – {metric}")
    plt.tight_layout()
    plt.savefig(f'../output/confusion_matrix_knn_{metric}.png')
    plt.close()

# 5. Zusammenfassung der kNN-Ergebnisse
print("\n📊 Zusammenfassung aller kNN GridSearch-Ergebnisse:")
for metric, res in results_knn.items():
    print(f"{metric:<15} | Score: {res['score_mean']:.4f} ± {res['score_std']:.4f} | Params: {res['params']}")

    
#################### Cross-Validated Confusion Matrix ####################

# 1. Wähle hier dein finales Modell (z.B. nach Accuracy)
best_model = results_knn['accuracy']['model']

# 2. Cross-validated Predictions (aggregiert über alle Folds)
y_pred_cv = cross_val_predict(best_model, X, y, cv=cv, n_jobs=-1)

# 3. Unnormierte Confusion Matrix berechnen
cm_cv = confusion_matrix(y, y_pred_cv, labels=all_labels)
print("🔢 Cross-Validated Confusion Matrix (counts):")
print(cm_cv)

# 4. Plotten und Speichern
disp_cv = ConfusionMatrixDisplay(confusion_matrix=cm_cv, display_labels=labels)
disp_cv.plot(cmap="Blues", xticks_rotation=45)
plt.title("Cross-Validated Confusion Matrix (counts)")
plt.tight_layout()
plt.savefig(f'../output/cm_cross_validated_counts.png')
plt.close()

# 5. (Optional) Normalisierte Version pro True-Label
cm_cv_norm = confusion_matrix(y, y_pred_cv, labels=all_labels, normalize='true')
print("📊 Cross-Validated Confusion Matrix (normalized per true label):")
print(cm_cv_norm)

disp_cv_norm = ConfusionMatrixDisplay(confusion_matrix=cm_cv_norm, display_labels=labels)
disp_cv_norm.plot(cmap="Blues", xticks_rotation=45)
plt.title("Cross-Validated Confusion Matrix (normalized)")
plt.tight_layout()
plt.savefig(f'../output/cm_cross_validated_norm.png')
plt.close()

####################### ROC_AUC curve #####################

# 1. Binarize die Test-Labels für One-vs-Rest
y_test_bin = label_binarize(y_test, classes=all_labels)  # shape: (n_samples, 5)
n_classes = y_test_bin.shape[1]

# 2. Wahrscheinlichkeiten des Modells abrufen
y_score = best_model.predict_proba(X_test)  # shape: (n_samples, 5)

# 3. FPR, TPR und AUC für jede Klasse berechnen
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_score[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])
    print(f"AUC für Klasse {labels[i]}: {roc_auc[i]:.3f}")  # z.B. AUC für Klasse LumA: 0.923

# 4. Mikro- und Makro-Average berechnen
# Mikro-Average über alle Klassen
fpr["micro"], tpr["micro"], _ = roc_curve(y_test_bin.ravel(), y_score.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
print(f"AUC (micro-average): {roc_auc['micro']:.3f}")

# Makro-Average: mittlere TPR über alle Klassen
all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
mean_tpr = np.zeros_like(all_fpr)
for i in range(n_classes):
    mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
mean_tpr /= n_classes
fpr["macro"] = all_fpr
tpr["macro"] = mean_tpr
roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
print(f"AUC (macro-average): {roc_auc['macro']:.3f}")

# 5. ROC-Kurven plotten
plt.figure()
# Einfache Farben, keine Manuelle Farbwahl
for i in range(n_classes):
    plt.plot(fpr[i], tpr[i],
             label=f"{labels[i]} (AUC = {roc_auc[i]:.3f})")
plt.plot(fpr["micro"], tpr["micro"],
         label=f"micro-average (AUC = {roc_auc['micro']:.3f})", linestyle=':', linewidth=2)
plt.plot(fpr["macro"], tpr["macro"],
         label=f"macro-average (AUC = {roc_auc['macro']:.3f})", linestyle='--', linewidth=2)

plt.plot([0, 1], [0, 1], 'k--', linewidth=1)  # Zufalls-Linie
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Multi-Class ROC-Kurven')
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig(f'../output/roc_auc_curve.png')
plt.close()

#######################Printing#############################
    
print("\n📋 Klassenspezifische Auswertung aller Modelle:")
for metric in metrics:
    print(f"\n📌 Metrik: {metric}")
    best_model = results_knn[metric]['model']
    best_model.fit(X_train, y_train)
    y_pred = best_model.predict(X_test)
    print(classification_report(y_test, y_pred, target_names=labels, labels=all_labels))
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1, 2, 3, 4])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(cmap="Blues", xticks_rotation=45)
    plt.title(f"Confusion Matrix – {metric}")
    plt.tight_layout()
    plt.savefig(f'../output/confusion_matrix_{metric}.png')
    plt.close()

# Zusammenfassung der Ergebnisse
print("\n📊 Zusammenfassung aller kNN GridSearch-Ergebnisse:")
for metric, res in results_knn.items():
    print(f"{metric:<15} | Score: {res['score_mean']:.4f} ± {res['score_std']:.4f} | Params: {res['params']}")
