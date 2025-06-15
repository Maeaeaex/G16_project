import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold, train_test_split, cross_val_predict, permutation_test_score, cross_val_score,  GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder, label_binarize
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_classif, VarianceThreshold
from sklearn.svm import SVC
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, auc, roc_auc_score)
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
from itertools import cycle
from sklearn.decomposition import PCA
from matplotlib.colors import ListedColormap

#Load and clean data
data = pd.read_csv("../data/dataset.csv").rename(columns={"Unnamed: 0": "Patient_ID"})
outcome = pd.read_csv("../data/outcome.csv").rename(columns={"Unnamed: 0": "Patient_ID"})

outcome_clean = outcome.dropna(subset=["BRCA_subtype"])
valid_patient_ids = outcome_clean["Patient_ID"]
data = data[data["Patient_ID"].isin(valid_patient_ids)]
outcome = outcome_clean
merged_data = pd.merge(data, outcome, on="Patient_ID")

X = merged_data.drop(columns=["Patient_ID", "BRCA_subtype"])
y = merged_data["BRCA_subtype"]

#Encode labels
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Plot class distribution
plt.figure(figsize=(8, 5))
sns.countplot(x=y, order=le.classes_)
plt.title("Class Distribution of BRCA Subtypes")
plt.xlabel("Subtype")
plt.ylabel("Number of Samples")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

n_classes = len(le.classes_)

#Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, stratify=y_encoded, random_state=20
)
print(np.bincount(y_train))
print(np.bincount(y_test))

#PCA for 95% variance
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_train)
pca = PCA()
pca.fit(X_scaled)
cum_var = np.cumsum(pca.explained_variance_ratio_)
n_components_95 = np.argmax(cum_var >= 0.95) + 1
print(f"Number of PCA components to retain 95% variance: {n_components_95}")

#Pipeline
base_pipeline = ImbPipeline([
    ('smote', SMOTE(random_state=20)),
    ('variance_filter', VarianceThreshold(threshold=0.0)),
    ('scaler', StandardScaler()),
    ('feature_selection', SelectKBest(score_func=f_classif)),
    ('pca', PCA()),
    ('svm', SVC(kernel='linear', probability=True, class_weight='balanced', random_state=20))
])

#Cross-validation with prediction
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=20)

# Hyperparameters
param_grid = [
    {
        'svm__C': [0.1, 1],
        'feature_selection__k': [1000, 2000, 5000],
        'pca__n_components': [100, 200, n_components_95]
    }]

grid_search = GridSearchCV(
    estimator=base_pipeline,
    param_grid=param_grid,
    cv=cv,
    scoring='roc_auc_ovr_weighted',
    n_jobs=-1,
    verbose=1
)

grid_search.fit(X_train, y_train)

print("\n--- Best Parameters from GridSearchCV ---")
print(grid_search.best_params_)
best_model = grid_search.best_estimator_

#Top 30 PCA Component Contributions
pca_fitted = best_model.named_steps['pca']
selector_fitted = best_model.named_steps['feature_selection']
var_filter_fitted = best_model.named_steps['variance_filter']

filtered_indices = var_filter_fitted.get_support(indices=True)
filtered_feature_names = X.columns[filtered_indices]

selected_indices = selector_fitted.get_support(indices=True)
selected_feature_names = filtered_feature_names[selected_indices]

components_df = pd.DataFrame(
    data=np.abs(pca_fitted.components_),
    columns=selected_feature_names
)

component_contributions = components_df.head(10).sum().sort_values(ascending=False)
pca_top_genes = component_contributions.head(30).reset_index()
pca_top_genes.columns = ["Gene", "Contribution"]

plt.figure(figsize=(10, 6))
sns.barplot(data=pca_top_genes, x="Contribution", y="Gene")
plt.title("Top 30 Genes by PCA Component Contribution")
plt.xlabel("Sum of Absolute Weights (Top 10 Components)")
plt.ylabel("Gene")
plt.tight_layout()
plt.show()

best_k = grid_search.best_params_['feature_selection__k']
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_train)

# Remove constant features
var_filter = VarianceThreshold(threshold=0.0)
X_var = var_filter.fit_transform(X_scaled)

# Fix k if too large
best_k = min(best_k, X_var.shape[1])

# Apply SelectKBest
selector = SelectKBest(score_func=f_classif, k=best_k)
selector.fit(X_var, y_train)

# Ensure NumPy array to avoid Pandas indexing issues
feature_names_after_var = np.array(X.columns[var_filter.get_support(indices=True)])
selected_indices = selector.get_support(indices=True)
selected_features = feature_names_after_var[selected_indices]

# Create DataFrame
feature_scores = selector.scores_
scores_df = pd.DataFrame({
    "Gene": selected_features,
    "F_score": feature_scores[selected_indices]
}).sort_values(by="F_score", ascending=False)

# Plot top 30 genes
plt.figure(figsize=(10, 6))
sns.barplot(data=scores_df.head(30), x="F_score", y="Gene")
plt.title("Top 30 Selected Genes by ANOVA F-Score (Before PCA)")
plt.xlabel("F-Score")
plt.ylabel("Gene")
plt.tight_layout()
plt.show()

y_val_pred = cross_val_predict(best_model, X_train, y_train, cv=cv)
y_val_proba = cross_val_predict(best_model, X_train, y_train, cv=cv, method='predict_proba')

# Cross-validated metrics with standard deviation
acc_scores = cross_val_score(best_model, X_train, y_train, cv=cv, scoring='accuracy')
prec_scores = cross_val_score(best_model, X_train, y_train, cv=cv, scoring='precision_weighted')
rec_scores = cross_val_score(best_model, X_train, y_train, cv=cv, scoring='recall_weighted')
f1_scores = cross_val_score(best_model, X_train, y_train, cv=cv, scoring='f1_weighted')
auc_scores = cross_val_score(best_model, X_train, y_train, cv=cv, scoring='roc_auc_ovr_weighted')

print("\n--- Cross-Validated Train Metrics (Mean ± Std) ---")
print(f"Accuracy:   {acc_scores.mean():.3f} ± {acc_scores.std():.3f}")
print(f"Precision:  {prec_scores.mean():.3f} ± {prec_scores.std():.3f}")
print(f"Recall:     {rec_scores.mean():.3f} ± {rec_scores.std():.3f}")
print(f"F1 Score:   {f1_scores.mean():.3f} ± {f1_scores.std():.3f}")
print(f"ROC AUC:    {auc_scores.mean():.3f} ± {auc_scores.std():.3f}")

#Per-class ROC AUC
y_train_bin = label_binarize(y_train, classes=np.arange(n_classes))
for i in range(n_classes):
    auc_i = roc_auc_score(y_train_bin[:, i], y_val_proba[:, i])
    print(f"ROC AUC for class {le.classes_[i]}: {auc_i:.3f}")

#Dummy classifier baseline
dummy_pipeline = Pipeline([
    ('dummy', DummyClassifier(strategy="most_frequent", random_state=20))
])
dummy_scores = cross_val_score(dummy_pipeline, X_train, y_train, cv=cv, scoring="accuracy")
print(f"\nDummy Classifier Accuracy: {dummy_scores.mean():.3f} ± {dummy_scores.std():.3f}")

#Nested cross-validation AUC
nested_scores = cross_val_score(best_model, X_train, y_train, cv=cv, scoring='roc_auc_ovr_weighted')
print(f"\nNested CV ROC AUC (macro): {nested_scores.mean():.3f} ± {nested_scores.std():.3f}")

#Final fit and test
y_test_pred = best_model.predict(X_test)
y_test_proba = best_model.predict_proba(X_test)

#Test Evaluation
test_acc = accuracy_score(y_test, y_test_pred)
test_prec = precision_score(y_test, y_test_pred, average="weighted", zero_division=0)
test_rec = recall_score(y_test, y_test_pred, average="weighted")
test_f1 = f1_score(y_test, y_test_pred, average="weighted")
test_auc = roc_auc_score(label_binarize(y_test, classes=np.arange(n_classes)), y_test_proba, average="macro", multi_class="ovr")

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

acc_scores = []
prec_scores = []
rec_scores = []
f1_scores = []
auc_scores = []

for train_idx, test_idx in skf.split(X_test, y_test):
    X_fold = X_test.iloc[test_idx]
    y_fold = y_test[test_idx]

    y_pred = best_model.predict(X_fold)
    y_proba = best_model.predict_proba(X_fold)

    acc_scores.append(accuracy_score(y_fold, y_pred))
    prec_scores.append(precision_score(y_fold, y_pred, average="weighted", zero_division=0))
    rec_scores.append(recall_score(y_fold, y_pred, average="weighted"))
    f1_scores.append(f1_score(y_fold, y_pred, average="weighted"))

    y_bin = label_binarize(y_fold, classes=np.arange(n_classes))
    auc_scores.append(roc_auc_score(y_bin, y_proba, average="macro", multi_class="ovr"))

print("\n--- Final Test Metrics ---")
print(f"Accuracy:   {test_acc:.3f} ± {np.std(acc_scores):.3f}")
print(f"Precision:  {test_prec:.3f} ± {np.std(prec_scores):.3f}")
print(f"Recall:     {test_rec:.3f} ± {np.std(rec_scores):.3f}")
print(f"F1 Score:   {test_f1:.3f} ± {np.std(f1_scores):.3f}")
print(f"ROC AUC:    {test_auc:.3f} ± {np.std(auc_scores):.3f}")

#Per-class AUC on test
y_test_bin = label_binarize(y_test, classes=np.arange(n_classes))
for i in range(n_classes):
    auc_i = roc_auc_score(y_test_bin[:, i], y_test_proba[:, i])
    print(f"ROC AUC for class {le.classes_[i]}: {auc_i:.3f}")
#Confusion Matrix
cm = confusion_matrix(y_test, y_test_pred)
class_names = le.classes_

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names,
            yticklabels=class_names)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('SVM Confusion Matrix')
plt.tight_layout()
plt.show()

#ROC Curve
y_test_bin = label_binarize(y_test, classes=np.arange(n_classes))
fpr, tpr, roc_auc = {}, {}, {}

for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_test_proba[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

plt.figure(figsize=(10, 6))
colors = cycle(['blue', 'orange', 'green', 'red', 'purple'])

for i, color in zip(range(n_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=2,
             label=f"Class {le.classes_[i]} (AUC = {roc_auc[i]:.2f})")

plt.plot([0, 1], [0, 1], 'k--', lw=2)
plt.xlim([-0.02, 1.02])
plt.ylim([-0.02, 1.02])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('SVM ROC Curves (Test Set)')
plt.legend(loc='lower right')
plt.grid(True)
plt.tight_layout()
plt.show()

"""
# Reduce to 2D using PCA (for visualization only)
pca = PCA(n_components=2)
X_test_scaled = StandardScaler().fit_transform(X_test)
X_test_pca = PCA(n_components=2).fit_transform(X_test_scaled)

# Train a new SVC on PCA-transformed test data (for boundary plotting)
svc_vis = SVC(kernel='linear', probability=False, random_state=20)
svc_vis.fit(X_test_pca, y_test)

# Create meshgrid for decision boundary
x_min, x_max = X_test_pca[:, 0].min() - 1, X_test_pca[:, 0].max() + 1
y_min, y_max = X_test_pca[:, 1].min() - 1, X_test_pca[:, 1].max() + 1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300),
                     np.linspace(y_min, y_max, 300))

# Predict labels across grid
Z = svc_vis.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Color map for 5 classes
cmap = ListedColormap(['royalblue', 'orange', 'lightgrey', 'salmon', 'darkred'])

plt.figure(figsize=(10, 6))
plt.contourf(xx, yy, Z, cmap=cmap, alpha=0.4)

# Plot test samples
scatter = plt.scatter(X_test_pca[:, 0], X_test_pca[:, 1], c=y_test, cmap=cmap, edgecolor='k')

# Legend mapping
handles = scatter.legend_elements()[0]
plt.legend(handles, le.classes_, title="Cancer Types", loc="upper left")

plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.title("SVC Decision Boundary Plot")
plt.tight_layout()
plt.show()
"""