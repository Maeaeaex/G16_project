#!/usr/bin/env python
# coding: utf-8

# In[187]:

import os







#get_ipython().run_line_magic('pwd', '')


# In[189]:


import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import accuracy_score


# In[190]:


df1 = pd.read_csv('../data/dataset.csv')


# In[191]:


df1.isna().sum().sum()


# In[192]:


df2 = pd.read_csv('../data/outcome.csv')
df2.isna().sum().sum()


# In[193]:


df_encoded = pd.get_dummies(df2.loc[:,'BRCA_subtype'], dtype=float, dummy_na=True)
#df_encoded
df3 = pd.concat([df1,df_encoded],axis = 1)
#print('we had the values of',df3.shape[0],'patients, but had to drop',dropped_rows,'now we have',df.shape[0],'remaining')
#we had the values of 1218 patients, but had to drop 262 now we have 956 remaining
df = df3[df3.iloc[:, -1]!=1]

#rows_to_drop = df3[df3.iloc[:, -1] == 1].index
#rows_to_drop
#df.drop(rows_to_drop, )

df = df.drop(df.columns[-1], axis=1)
df


# In[194]:


df_patient = df.iloc[:,0]
df = df.drop(df.columns[0], axis=1)
#df


# In[ ]:





# In[195]:


#split into test and train dataset.

# Separate features (X) and target (y)
y = df[['Basal','Her2','LumA','LumB','Normal']]      # Target variable (dependent variable)
X = df.drop( ['Basal','Her2','LumA','LumB','Normal'] , axis=1)  # Features (independent variables)
#print(y, X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[196]:


#create scalar
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


#create model
clf = LogisticRegression(max_iter=1000, random_state=2025)
#make multi outcome model
multi_output_model = MultiOutputClassifier(clf)
#fit model
multi_output_model.fit(X_train, y_train)


# In[197]:


import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import accuracy_score
#load the data
df1 = pd.read_csv('../data/dataset.csv')
df2 = pd.read_csv('../data/outcome.csv')
df_encoded = pd.get_dummies(df2.loc[:,'BRCA_subtype'], dtype=float, dummy_na=True)
#add dataframes togetehr and drop the missing data
df3 = pd.concat([df1,df_encoded],axis = 1)
df = df3[df3.iloc[:, -1]!=1]
df = df.drop(df.columns[-1], axis=1)
#drop the patient numbers so we can use the model
df_patient = df.iloc[:,0]
df = df.drop(df.columns[0], axis=1)
#split into test and train dataset.
# Separate features (X) and target (y)
y = df[['Basal','Her2','LumA','LumB','Normal']]      # Target variable (dependent variable)
X = df.drop( ['Basal','Her2','LumA','LumB','Normal'] , axis=1)  # Features (independent variables)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#create scalar
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
#create model
clf = LogisticRegression(max_iter=1000, random_state=2025)
#make multi outcome model
multi_output_model = MultiOutputClassifier(clf)
#fit model
multi_output_model.fit(X_train, y_train)


# In[ ]:


#lasso method would be required here


# In[158]:


from sklearn.metrics import accuracy_score, classification_report

# Predict using the test data
y_pred = multi_output_model.predict(X_test)

# Evaluate the model using accuracy for each label
for i, label in enumerate(y.columns):
    acc = accuracy_score(y_test[label], y_pred[:, i])
    print(f"Accuracy for {label}: {acc:.2f}")
    print(f"Classification report for {label}:\n{classification_report(y_test[label], y_pred[:, i])}\n")

# Optional: If you want an overall accuracy metric (not very meaningful for multilabel usually)
overall_accuracy = (y_pred == y_test.values).mean()
print(f"Overall label-wise accuracy: {overall_accuracy:.2f}")


# In[198]:


from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Make predictions
y_pred = multi_output_model.predict(X_test)

# Calculate metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='macro', zero_division=0)
recall = recall_score(y_test, y_pred, average='macro', zero_division=0)
f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)

# Print results
print("Accuracy: {:.4f}".format(accuracy))
print("Precision (macro): {:.4f}".format(precision))
print("Recall (macro): {:.4f}".format(recall))
print("F1 Score (macro): {:.4f}".format(f1))


# In[159]:


# Get the feature names
feature_names = X.columns

# Loop through each label's model
for i, label in enumerate(y.columns):
    # Get the coefficients of the model for this label
    coefs = multi_output_model.estimators_[i].coef_[0]  # shape: (n_features,)
    
    # Create a DataFrame of features and their coefficients
    coef_df = pd.DataFrame({
        'Feature': feature_names,
        'Coefficient': coefs,
        'Abs_Coefficient': np.abs(coefs)
    }).sort_values(by='Abs_Coefficient', ascending=False)
    
    print(f"\nTop 10 features for {label}:")
    print(coef_df.head(10))


# In[160]:


import seaborn as sns

top_n = 10
sns.barplot(x='Abs_Coefficient', y='Feature', data=coef_df.head(top_n))
plt.title(f'Top {top_n} Features for {label}')
plt.xlabel('Absolute Coefficient')
plt.ylabel('Feature')
plt.show()


# In[161]:


from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Loop through each label and plot its confusion matrix
for i, label in enumerate(y.columns):
    # Get true and predicted values for the current label
    y_true_label = y_test[label]
    y_pred_label = y_pred[:, i]

    # Generate confusion matrix
    cm = confusion_matrix(y_true_label, y_pred_label)
    
    # Plot confusion matrix
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0, 1])
    disp.plot()
    plt.title(f'Confusion Matrix for {label}')
    plt.show()

   # print(f"Confusion matrix for {label}:\n{cm}")


# In[162]:


import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

n_labels = len(y.columns)
cols = 3  # Number of columns in the grid
rows = (n_labels + cols - 1) // cols  # Compute number of rows

fig, axs = plt.subplots(rows, cols, figsize=(15, 5 * rows))

for i, label in enumerate(y.columns):
    row = i // cols
    col = i % cols
    
    # Compute confusion matrix
    y_true_label = y_test[label]
    y_pred_label = y_pred[:, i]
    cm = confusion_matrix(y_true_label, y_pred_label)
    
    # Plot
    ax = axs[row, col] if rows > 1 else axs[col]
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0, 1])
    disp.plot(ax=ax, values_format='d')
    ax.set_title(f'{label}')

# Turn off any unused subplots
for j in range(i + 1, rows * cols):
    fig.delaxes(axs.flatten()[j])

plt.tight_layout()
plt.suptitle('Confusion Matrices for All Subtypes', fontsize=16, y=1.02)
plt.show()


# In[184]:


#Cross-label Confusion Matrix
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Convert y_test and y_pred to numpy arrays for easier manipulation
true_labels = y_test.values
pred_labels = y_pred

# Get label names
labels = y.columns.tolist()
n_labels = len(labels)

# Initialize an empty confusion matrix (label-to-label)
cross_confusion = np.zeros((n_labels, n_labels), dtype=int)

# Build the matrix
for true, pred in zip(true_labels, pred_labels):
    true_indices = np.where(true == 1)[0]
    pred_indices = np.where(pred == 1)[0]
    
    for t in true_indices:
        for p in pred_indices:
            cross_confusion[t, p] += 1

# Convert to DataFrame for better display
cross_conf_df = pd.DataFrame(cross_confusion, index=labels, columns=labels)

# Plot the heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(cross_conf_df, annot=True, fmt='d', cmap='cubehelix_r')
plt.title('Cross-label Confusion Matrix (True vs. Predicted Labels)')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.tight_layout()
plt.show()


# In[185]:


from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pandas as pd

# Create empty list to collect metrics
metrics_summary = []

# Loop through each label
for i, label in enumerate(y.columns):
    y_true = y_test[label]
    y_pred_label = y_pred[:, i]

    metrics_summary.append({
        'Label': label,
        'Accuracy': accuracy_score(y_true, y_pred_label),
        'Precision': precision_score(y_true, y_pred_label, zero_division=0),
        'Recall': recall_score(y_true, y_pred_label, zero_division=0),
        'F1 Score': f1_score(y_true, y_pred_label, zero_division=0)
    })

# Convert to DataFrame
metrics_df = pd.DataFrame(metrics_summary)
metrics_df.set_index('Label', inplace=True)
metrics_df = metrics_df.round(3)  # Optional: round for clarity

# Display table
print(metrics_df)


# In[186]:


# Predict probabilities instead of binary labels
y_proba = multi_output_model.predict_proba(X_test)

# Convert list of arrays to 2D array (samples × labels)
y_score = np.array([proba[:, 1] for proba in y_proba]).T  # shape: (n_samples, n_labels)

from sklearn.metrics import roc_auc_score

# Calculate ROC AUC per label
roc_auc_per_label = {}
for i, label in enumerate(y.columns):
    try:
        auc = roc_auc_score(y_test[label], y_score[:, i])
    except ValueError:
        auc = np.nan  # In case there are no positives or negatives
    roc_auc_per_label[label] = auc

# Convert to DataFrame
import pandas as pd
roc_df = pd.DataFrame.from_dict(roc_auc_per_label, orient='index', columns=['ROC AUC'])

# Optional: Add overall averages
roc_df.loc['Macro avg'] = roc_auc_score(y_test, y_score, average='macro')
roc_df.loc['Micro avg'] = roc_auc_score(y_test, y_score, average='micro')

# Round and print
roc_df = roc_df.round(3)
print(roc_df)


# In[199]:


#ROC AUC
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt

# Get probability predictions instead of hard labels
y_proba = multi_output_model.predict_proba(X_test)

# Convert list of arrays to 2D array (each column is for one class)
y_proba = np.array([prob[:, 1] for prob in y_proba]).T  # Shape: (n_samples, n_classes)

# Compute ROC AUC for each label
for i, label in enumerate(y.columns):
    auc = roc_auc_score(y_test.iloc[:, i], y_proba[:, i])
    fpr, tpr, _ = roc_curve(y_test.iloc[:, i], y_proba[:, i])

    plt.figure()
    plt.plot(fpr, tpr, label=f'ROC curve (AUC = {auc:.2f})')
    plt.plot([0, 1], [0, 1], 'k--', label='Random chance')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve for {label}')
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.show()


# In[201]:


from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt

# Get predicted probabilities
y_proba = multi_output_model.predict_proba(X_test)

# Extract the probability of the positive class for each label
y_proba = np.array([prob[:, 1] for prob in y_proba]).T  # Shape: (n_samples, n_classes)

# Plot all ROC curves in one figure
plt.figure(figsize=(8, 6))

# Loop over each class
for i, label in enumerate(y.columns):
    fpr, tpr, _ = roc_curve(y_test.iloc[:, i], y_proba[:, i])
    auc = roc_auc_score(y_test.iloc[:, i], y_proba[:, i])
    plt.plot(fpr, tpr, label=f'{label} (AUC = {auc:.2f})')

# Plot random chance line
plt.plot([0, 1], [0, 1], 'k--', label='Random chance')

# Configure the plot
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves for Each BRCA Subtype')
plt.legend(loc='lower right')
plt.grid(True)
plt.tight_layout()
plt.show()


from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score, accuracy_score, confusion_matrix
import numpy as np
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold

# Number of cross-validation splits
n_splits = 5
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

# We need to stratify on a single label (e.g., argmax of one-hot encoded labels)
y_stratify = y.values.argmax(axis=1)

# Store metrics for each fold
metrics_list = []

# Perform cross-validation
for train_idx, test_idx in skf.split(X, y_stratify):
    X_train_fold, X_test_fold = X.iloc[train_idx], X.iloc[test_idx]
    y_train_fold, y_test_fold = y.iloc[train_idx], y.iloc[test_idx]

    # Scale the features
    X_train_scaled = scaler.fit_transform(X_train_fold)
    X_test_scaled = scaler.transform(X_test_fold)

    # Clone the model to avoid data leakage
    model = clone(multi_output_model)
    model.fit(X_train_scaled, y_train_fold)
    y_pred_fold = model.predict(X_test_scaled)

    # Compute the metrics for the whole model (macro averaged)
    accuracy = accuracy_score(y_test_fold, y_pred_fold)
    precision = precision_score(y_test_fold, y_pred_fold, average='macro', zero_division=0)
    recall = recall_score(y_test_fold, y_pred_fold, average='macro', zero_division=0)
    f1 = f1_score(y_test_fold, y_pred_fold, average='macro', zero_division=0)

    # Compute ROC AUC for the entire set of labels (macro average)
    y_proba = model.predict_proba(X_test_scaled)
    y_score = np.array([proba[:, 1] for proba in y_proba]).T  # shape: (n_samples, n_labels)
    roc_auc = roc_auc_score(y_test_fold, y_score, average='macro', multi_class='ovr')

    # Compute Sensitivity (recall) and Specificity
    sensitivities = []
    specificities = []

    for i in range(y_test_fold.shape[1]):  # Iterate through each label
        tn, fp, fn, tp = confusion_matrix(y_test_fold.iloc[:, i], y_pred_fold[:, i], labels=[0, 1]).ravel()
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        sensitivities.append(sensitivity)
        specificities.append(specificity)

    # Store fold metrics
    metrics_list.append({
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1': f1,
        'ROC AUC': roc_auc,
        'Sensitivity': np.mean(sensitivities),  # Average sensitivity
        'Specificity': np.mean(specificities)   # Average specificity
    })

# Convert list of metrics into a DataFrame
metrics_df = pd.DataFrame(metrics_list)

# Compute mean and standard deviation for each metric across folds
metrics_summary = metrics_df.describe().loc[['mean', 'std']].T

# Display results
print("\nOverall Model Performance (Mean ± Std):")
print(metrics_summary.round(3))
