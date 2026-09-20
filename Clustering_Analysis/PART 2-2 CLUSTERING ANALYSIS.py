# ============================================================
# PART 2-2: CLUSTERING ANALYSIS
# ============================================================

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA


# ============================================================
# 1. IMPORTING DATASET
# ============================================================

file_path = r"C:\Users\shaffan\Downloads\Shaffan\Dataset_ATS_v2.csv"

df = pd.read_csv(file_path)

print("Dataset loaded successfully.")
print("Original dataset shape:", df.shape)


# ============================================================
# 2. CLEANING DATASET
# ============================================================

# duplicate records Removal 
duplicate_count = df.duplicated().sum()

print("\nDuplicate records:", duplicate_count)

df = df.drop_duplicates().reset_index(drop=True)

print("Dataset shape after duplicate removal:", df.shape)


# ============================================================
# 3. DATASET PREPARATION FOR CLUSTERING
# ============================================================

# Churn is the target variable.
# It is NOT used for creating the clusters because
# K-Means clustering is an unsupervised technique.

X_cluster = df.drop(columns=["Churn"])


# Numerical and categorical features Identification
numerical_features = X_cluster.select_dtypes(
    include=["int64", "float64"]
).columns.tolist()

categorical_features = X_cluster.select_dtypes(
    include=["object", "category"]
).columns.tolist()


print("\nNumerical features:")
print(numerical_features)

print("\nCategorical features:")
print(categorical_features)


# ============================================================
# 4. CATEGORICAL VARIABLES ENCODING AND SCALING NUMERICAL VARIABLES
# ============================================================

preprocessor = ColumnTransformer(
    transformers=[
        (
            "numerical",
            StandardScaler(),
            numerical_features
        ),
        (
            "categorical",
            OneHotEncoder(
                drop="first",
                handle_unknown="ignore",
                sparse_output=False
            ),
            categorical_features
        )
    ]
)


X_cluster_processed = preprocessor.fit_transform(
    X_cluster
)

print(
    "\nProcessed clustering data shape:",
    X_cluster_processed.shape
)


# ============================================================
# 5. ELBOW METHOD
# ============================================================

k_values = range(1, 11)

inertia_values = []

for k in k_values:

    kmeans = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    kmeans.fit(X_cluster_processed)

    inertia_values.append(
        kmeans.inertia_
    )


# ============================================================
# 6. ELBOW METHOD RESULTS
# ============================================================

print("\n============================================================")
print("ELBOW METHOD RESULTS")
print("============================================================")

for k, inertia in zip(
    k_values,
    inertia_values
):

    print(
        f"K = {k} | Inertia = {inertia:.2f}"
    )


# ============================================================
# 7. ELBOW METHOD VISUALIZATION
# ============================================================

plt.figure(figsize=(9, 6))

plt.plot(
    k_values,
    inertia_values,
    marker="o"
)

plt.xlabel("Number of Clusters (K)")
plt.ylabel(
    "Inertia (Within-Cluster Sum of Squares)"
)

plt.title(
    "Elbow Method for Optimal Number of Clusters"
)

plt.xticks(list(k_values))

plt.grid(True)

plt.tight_layout()

plt.show()


# ============================================================
# 8. OPTIMAL NUMBER OF CLUSTERS SELECTIONN
# ============================================================

# Based on the Elbow Method results, K = 3 is selected.

optimal_k = 3

print(
    "\nOptimal number of clusters:",
    optimal_k
)


# ============================================================
# 9. TRAINING FINAL K-MEANS MODEL
# ============================================================

kmeans_model = KMeans(
    n_clusters=optimal_k,
    random_state=42,
    n_init=10
)

cluster_labels = kmeans_model.fit_predict(
    X_cluster_processed
)


# Add cluster assignments to the dataset
df["Cluster"] = cluster_labels


# ============================================================
# 10. CLUSTER SIZE
# ============================================================

cluster_sizes = (
    df["Cluster"]
    .value_counts()
    .sort_index()
)

print("\n============================================================")
print("CLUSTER SIZES")
print("============================================================")

print(cluster_sizes)


# ============================================================
# 11. PCA FOR VISUALIZATION
# ============================================================

pca = PCA(
    n_components=2,
    random_state=42
)

X_pca = pca.fit_transform(
    X_cluster_processed
)

explained_variance = (
    pca.explained_variance_ratio_
)

print("\nPCA explained variance:")
print(explained_variance)

print(
    "Total variance represented by 2 components:",
    round(
        explained_variance.sum() * 100,
        2
    ),
    "%"
)


# ============================================================
# 12. PCA DATAFRAME
# ============================================================

pca_df = pd.DataFrame(
    X_pca,
    columns=[
        "Principal Component 1",
        "Principal Component 2"
    ]
)

pca_df["Cluster"] = cluster_labels


# ============================================================
# 13. VISUALIZING K-MEANS CLUSTERS
# ============================================================

plt.figure(figsize=(10, 7))

for cluster in sorted(
    pca_df["Cluster"].unique()
):

    cluster_data = pca_df[
        pca_df["Cluster"] == cluster
    ]

    plt.scatter(
        cluster_data["Principal Component 1"],
        cluster_data["Principal Component 2"],
        label=f"Cluster {cluster}",
        alpha=0.6
    )


plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")

plt.title(
    "K-Means Customer Segmentation (K = 3)"
)

plt.legend(
    title="Cluster"
)

plt.grid(True)

plt.tight_layout()

plt.show()


# ============================================================
# 14. NUMERICAL CLUSTER PROFILES
# ============================================================

print("\n============================================================")
print("NUMERICAL CLUSTER PROFILES")
print("============================================================")

cluster_numeric_profile = (
    df.groupby("Cluster")[
        numerical_features
    ].mean()
)

print(
    cluster_numeric_profile.round(2)
)


# ============================================================
# 15. CATEGORICAL CLUSTER PROFILES
# ============================================================

print("\n============================================================")
print("CATEGORICAL CLUSTER PROFILES")
print("============================================================")

for column in categorical_features:

    print(
        f"\n--- {column} (%) ---"
    )

    profile = (
        pd.crosstab(
            df["Cluster"],
            df[column],
            normalize="index"
        ) * 100
    )

    print(
        profile.round(2)
    )


# ============================================================
# 16. CALCULATING CHURN RATE BY CLUSTER
# ============================================================

df["Churn_Binary"] = df["Churn"].map({
    "No": 0,
    "Yes": 1
})

cluster_churn = (
    df.groupby("Cluster")[
        "Churn_Binary"
    ]
    .agg(["count", "mean"])
)

cluster_churn["Churn_Rate_%"] = (
    cluster_churn["mean"] * 100
)

cluster_churn = cluster_churn.drop(
    columns=["mean"]
)

print("\n============================================================")
print("CHURN RATE BY CLUSTER")
print("============================================================")

print(
    cluster_churn.round(2)
)


# ============================================================
# 17. CUSTOMER SEGMENTS LABELING
# ============================================================

cluster_names = {
    0: "Loyal Established Customers",
    1: "Newer / Developing Customers",
    2: "Senior High-Risk Customers"
}

df["Cluster_Label"] = (
    df["Cluster"].map(cluster_names)
)


# ============================================================
# 18. FINAL CUSTOMER SEGMENT SUMMARY
# ============================================================

cluster_summary = (
    df.groupby(
        ["Cluster", "Cluster_Label"]
    )
    .agg(
        Customers=("Cluster", "size"),
        Average_Tenure=("tenure", "mean"),
        Average_Monthly_Charges=(
            "MonthlyCharges",
            "mean"
        ),
        Churn_Rate=(
            "Churn_Binary",
            "mean"
        )
    )
    .reset_index()
)

cluster_summary["Churn_Rate"] = (
    cluster_summary["Churn_Rate"] * 100
)


print("\n============================================================")
print("FINAL CUSTOMER SEGMENT SUMMARY")
print("============================================================")

print(
    cluster_summary.round(2)
)


# ============================================================
# 19. ASSIGNING SAMPLE CLUSTER
# ============================================================

print("\n============================================================")
print("SAMPLE CLUSTER ASSIGNMENTS")
print("============================================================")

print(
    df[
        [
            "gender",
            "SeniorCitizen",
            "Dependents",
            "tenure",
            "PhoneService",
            "MultipleLines",
            "InternetService",
            "Contract",
            "MonthlyCharges",
            "Churn",
            "Cluster",
            "Cluster_Label"
        ]
    ].head(20)
)