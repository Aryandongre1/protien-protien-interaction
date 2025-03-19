import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
ppi_data = pd.read_csv("string_ppi.csv")
print(ppi_data.head())
gene_exp = pd.read_csv("encode_gene_expression.csv")

# Normalize expression values
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
gene_exp.iloc[:, 1:] = scaler.fit_transform(gene_exp.iloc[:, 1:])  # Scale expression values

print(gene_exp.head())
 Feature Engineering (Combine PPI + Gene Expression Data)
# Merge datasets based on protein names
ppi_features = ppi_data.merge(gene_exp, left_on="Protein A", right_on="Gene")
ppi_features = ppi_features.merge(gene_exp, left_on="Protein B", right_on="Gene", suffixes=("_A", "_B"))

# Drop unnecessary columns
ppi_features = ppi_features.drop(columns=["Protein A", "Protein B", "Gene_A", "Gene_B"])

# Labels (1 = Known Interaction, 0 = No Interaction)
ppi_features["Label"] = (ppi_features["Interaction Score"] > 0.8).astype(int)

print(ppi_features.head())

Train Random Forest Model for PPI Classification
# Prepare training data
X = ppi_features.drop(columns=["Interaction Score", "Label"])
y = ppi_features["Label"]

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Random Forest model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Predict and evaluate
y_pred = rf_model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))


This project focuses on predicting Protein-Protein Interactions (PPIs) using machine learning models. By integrating gene expression data from the ENCODE database with known PPIs, we train a Random Forest classifier to classify whether two proteins interact.
