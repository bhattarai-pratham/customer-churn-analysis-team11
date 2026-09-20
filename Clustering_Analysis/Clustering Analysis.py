# ============================================================
# PART 2-2: CLUSTERING ANALYSIS
# ============================================================


# ------------------------------------------------------------
# IMPORTING REQUIRED LIBRARIES
# ------------------------------------------------------------

import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline


# ============================================================
# 1. SETTING FILE PATHS AND OUTPUT FOLDER
# ============================================================

file_path = r"C:\Users\shaffan\Downloads\Shaffan\Dataset_ATS_v2.csv"

output_folder = r"C:\Users\shaffan\Downloads\Shaffan\Clustering_Analysis"

os.makedirs(
    output_folder,
    exist_ok=True
)

print("Output folder:")
print(output_folder)


# ============================================================
# 2. IMPORTING DATASET
# ============================================================

df = pd.read_csv(file_path)

print("\nDataset loaded successfully.")
print("Original dataset shape:", df.shape)


# ============================================================
# 3. CHECKING AND CLEANING DATASET
# ============================================================

print("\nMissing values before cleaning:")
print(df.isnull().sum())


# Identifying categorical columns directly from the dataset
categorical_columns_all = [
    column
    for column in df.columns
    if df[column].dtype == "object"
    or str(df[column].dtype) == "category"
    or str(df[column].dtype) == "string"
]


# Handling missing numerical values
numerical_columns_all = [
    column
    for column in df.columns
    if column not in categorical_columns_all
]


for column in numerical_columns_all:

    if df[column].isnull().any():

        df[column] = df[column].fillna(
            df[column].median()
        )


# Handling missing categorical values
for column in categorical_columns_all:

    if df[column].isnull().any():

        df[column] = df[column].fillna(
            df[column].mode()[0]
        )


print("\nMissing values after cleaning:")
print(df.isnull().sum())


# ============================================================
# 4. REMOVING DUPLICATE RECORDS
# ============================================================

duplicate_count = df.duplicated().sum()

print(
    "\nDuplicate records:",
    duplicate_count
)


df = df.drop_duplicates().reset_index(
    drop=True
)


print(
    "Dataset shape after duplicate removal:",
    df.shape
)


# ============================================================
# 5. PREPARING DATASET FOR CLUSTERING
# ============================================================

# Churn is the target variable.
# It is excluded from clustering because K-Means
# is an unsupervised learning technique.

X_cluster = df.drop(
    columns=["Churn"]
)


# ============================================================
# 6. IDENTIFYING NUMERICAL AND CATEGORICAL FEATURES
# ============================================================

numerical_features = [
    column
    for column in X_cluster.columns
    if pd.api.types.is_numeric_dtype(
        X_cluster[column]
    )
]


categorical_features = [
    column
    for column in X_cluster.columns
    if column not in numerical_features
]


print("\nNumerical features:")
print(numerical_features)


print("\nCategorical features:")
print(categorical_features)


# ============================================================
# 7. ENCODING CATEGORICAL VARIABLES AND
#    SCALING NUMERICAL VARIABLES
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
# 8. APPLYING ELBOW METHOD
# ============================================================

k_values = range(
    1,
    11
)

inertia_values = []


for k in k_values:

    kmeans = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    kmeans.fit(
        X_cluster_processed
    )

    inertia_values.append(
        kmeans.inertia_
    )


# ============================================================
# 9. DISPLAYING ELBOW METHOD RESULTS
# ============================================================

print(
    "\n============================================================"
)

print(
    "ELBOW METHOD RESULTS"
)

print(
    "============================================================"
)


for k, inertia in zip(
    k_values,
    inertia_values
):

    print(
        f"K = {k} | Inertia = {inertia:.2f}"
    )


# ============================================================
# 10. SAVING ELBOW METHOD RESULTS
# ============================================================

elbow_results = pd.DataFrame(
    {
        "Number_of_Clusters": list(k_values),
        "Inertia": inertia_values
    }
)


elbow_results.to_csv(
    os.path.join(
        output_folder,
        "Elbow_Method_Results.csv"
    ),
    index=False
)


# ============================================================
# 11. VISUALIZING ELBOW METHOD
# ============================================================

plt.figure(
    figsize=(9, 6)
)


plt.plot(
    k_values,
    inertia_values,
    marker="o"
)


plt.xlabel(
    "Number of Clusters (K)"
)

plt.ylabel(
    "Inertia (Within-Cluster Sum of Squares)"
)

plt.title(
    "Elbow Method for Optimal Number of Clusters"
)

plt.xticks(
    list(k_values)
)

plt.grid(
    True
)

plt.tight_layout()


plt.savefig(
    os.path.join(
        output_folder,
        "Elbow_Method.png"
    ),
    dpi=300,
    bbox_inches="tight"
)


plt.show()


# ============================================================
# 12. SELECTING OPTIMAL NUMBER OF CLUSTERS
# ============================================================

# Based on the Elbow Method, K = 3 is selected.

optimal_k = 3


print(
    "\nOptimal number of clusters:",
    optimal_k
)


# ============================================================
# 13. DOCUMENTING OPTIMAL NUMBER OF CLUSTERS
# ============================================================

with open(
    os.path.join(
        output_folder,
        "Optimal_Number_of_Clusters.txt"
    ),
    "w",
    encoding="utf-8"
) as file:

    file.write(
        "OPTIMAL NUMBER OF CLUSTERS\n"
    )

    file.write(
        "==========================\n\n"
    )

    file.write(
        "Method: Elbow Method\n"
    )

    file.write(
        f"Selected optimal number of clusters: K = {optimal_k}\n\n"
    )

    file.write(
        "The Elbow Method was evaluated using K values "
        "from 1 to 10. The resulting inertia values were "
        "examined to determine the point at which the "
        "reduction in within-cluster sum of squares began "
        "to decrease progressively. Based on the resulting "
        "elbow plot, K = 3 was selected for the final "
        "K-Means clustering model.\n"
    )


# ============================================================
# 14. TRAINING FINAL K-MEANS MODEL
# ============================================================

kmeans_model = KMeans(
    n_clusters=optimal_k,
    random_state=42,
    n_init=10
)


cluster_labels = kmeans_model.fit_predict(
    X_cluster_processed
)


# Adding cluster assignments
df["Cluster"] = cluster_labels


print(
    "\nK-Means model trained successfully."
)


# ============================================================
# 15. SAVING TRAINED K-MEANS MODEL
# ============================================================

joblib.dump(
    kmeans_model,
    os.path.join(
        output_folder,
        "KMeans_Model.pkl"
    )
)


print(
    "Trained K-Means model saved successfully."
)


# ============================================================
# 16. CREATING AND SAVING K-MEANS PIPELINE
# ============================================================

kmeans_pipeline = Pipeline(
    steps=[

        (
            "preprocessing",
            preprocessor
        ),

        (
            "kmeans",
            KMeans(
                n_clusters=optimal_k,
                random_state=42,
                n_init=10
            )
        )
    ]
)


kmeans_pipeline.fit(
    X_cluster
)


joblib.dump(
    kmeans_pipeline,
    os.path.join(
        output_folder,
        "KMeans_Pipeline.pkl"
    )
)


print(
    "K-Means pipeline saved successfully."
)


# ============================================================
# 17. CALCULATING CLUSTER SIZES
# ============================================================

cluster_sizes = (
    df["Cluster"]
    .value_counts()
    .sort_index()
)


print(
    "\n============================================================"
)

print(
    "CLUSTER SIZES"
)

print(
    "============================================================"
)


print(
    cluster_sizes
)


# ============================================================
# 18. APPLYING PCA FOR CLUSTER VISUALIZATION
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


print(
    "\nPCA explained variance:"
)

print(
    explained_variance
)


print(
    "Total variance represented by 2 components:",
    round(
        explained_variance.sum() * 100,
        2
    ),
    "%"
)


# ============================================================
# 19. CREATING PCA DATAFRAME
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
# 20. LABELING RESULTING CUSTOMER CLUSTERS
# ============================================================

cluster_names = {

    0: "Loyal Established Customers",

    1: "Newer / Developing Customers",

    2: "Senior High-Risk Customers"
}


df["Cluster_Label"] = (
    df["Cluster"].map(
        cluster_names
    )
)


pca_df["Cluster_Label"] = (
    pca_df["Cluster"].map(
        cluster_names
    )
)


# ============================================================
# 21. VISUALIZING AND LABELING K-MEANS CLUSTERS
# ============================================================

plt.figure(
    figsize=(10, 7)
)


for cluster in sorted(
    pca_df["Cluster"].unique()
):

    cluster_data = pca_df[
        pca_df["Cluster"] == cluster
    ]


    plt.scatter(
        cluster_data[
            "Principal Component 1"
        ],

        cluster_data[
            "Principal Component 2"
        ],

        label=cluster_names[
            cluster
        ],

        alpha=0.6
    )


plt.xlabel(
    "Principal Component 1"
)

plt.ylabel(
    "Principal Component 2"
)

plt.title(
    "K-Means Customer Segmentation (K = 3)"
)

plt.legend(
    title="Customer Segment"
)

plt.grid(
    True
)

plt.tight_layout()


plt.savefig(
    os.path.join(
        output_folder,
        "KMeans_Clustering.png"
    ),
    dpi=300,
    bbox_inches="tight"
)


plt.show()


# ============================================================
# 22. CALCULATING NUMERICAL CLUSTER PROFILES
# ============================================================

print(
    "\n============================================================"
)

print(
    "NUMERICAL CLUSTER PROFILES"
)

print(
    "============================================================"
)


cluster_numeric_profile = (
    df.groupby(
        "Cluster"
    )[
        numerical_features
    ].mean()
)


print(
    cluster_numeric_profile.round(2)
)


cluster_numeric_profile.to_csv(
    os.path.join(
        output_folder,
        "Numerical_Cluster_Profiles.csv"
    )
)


# ============================================================
# 23. CALCULATING CATEGORICAL CLUSTER PROFILES
# ============================================================

print(
    "\n============================================================"
)

print(
    "CATEGORICAL CLUSTER PROFILES"
)

print(
    "============================================================"
)


categorical_profile_tables = []


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


    profile_for_export = (
        profile.reset_index()
    )


    profile_for_export.insert(
        1,
        "Feature",
        column
    )


    categorical_profile_tables.append(
        profile_for_export
    )


if categorical_profile_tables:

    categorical_profiles = pd.concat(
        categorical_profile_tables,
        ignore_index=True
    )


    categorical_profiles.to_csv(
        os.path.join(
            output_folder,
            "Categorical_Cluster_Profiles.csv"
        ),
        index=False
    )


# ============================================================
# 24. CALCULATING CHURN RATE BY CLUSTER
# ============================================================

df["Churn_Binary"] = df["Churn"].map(
    {
        "No": 0,
        "Yes": 1
    }
)


cluster_churn = (
    df.groupby(
        "Cluster"
    )[
        "Churn_Binary"
    ]
    .agg(
        Customer_Count="count",
        Churn_Rate="mean"
    )
)


cluster_churn["Churn_Rate"] = (
    cluster_churn["Churn_Rate"] * 100
)


print(
    "\n============================================================"
)

print(
    "CHURN RATE BY CLUSTER"
)

print(
    "============================================================"
)


print(
    cluster_churn.round(2)
)


cluster_churn.round(2).to_csv(
    os.path.join(
        output_folder,
        "Churn_Rate_By_Cluster.csv"
    )
)


# ============================================================
# 25. CREATING FINAL CUSTOMER SEGMENT SUMMARY
# ============================================================

cluster_summary = (
    df.groupby(
        [
            "Cluster",
            "Cluster_Label"
        ]
    )
    .agg(

        Customers=(
            "Cluster",
            "size"
        ),

        Average_Tenure=(
            "tenure",
            "mean"
        ),

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


print(
    "\n============================================================"
)

print(
    "FINAL CUSTOMER SEGMENT SUMMARY"
)

print(
    "============================================================"
)


print(
    cluster_summary.round(2)
)


cluster_summary.round(2).to_csv(
    os.path.join(
        output_folder,
        "Cluster_Summary.csv"
    ),
    index=False
)


# ============================================================
# 26. SAVING COMPLETE CLUSTERED DATASET
# ============================================================

df.to_csv(
    os.path.join(
        output_folder,
        "Clustered_Dataset.csv"
    ),
    index=False
)


# ============================================================
# 27. DOCUMENTING CUSTOMER SEGMENT LABELS
# ============================================================

with open(
    os.path.join(
        output_folder,
        "Cluster_Labels.txt"
    ),
    "w",
    encoding="utf-8"
) as file:

    file.write(
        "CUSTOMER CLUSTER LABELS\n"
    )

    file.write(
        "=======================\n\n"
    )


    for cluster, label in cluster_names.items():

        file.write(
            f"Cluster {cluster}: {label}\n"
        )

        file.write(
            "----------------------------------------\n"
        )


        summary_row = cluster_summary[
            cluster_summary["Cluster"] == cluster
        ]


        if not summary_row.empty:

            row = summary_row.iloc[0]


            file.write(
                f"Customers: "
                f"{int(row['Customers'])}\n"
            )

            file.write(
                f"Average tenure: "
                f"{row['Average_Tenure']:.2f}\n"
            )

            file.write(
                f"Average monthly charges: "
                f"{row['Average_Monthly_Charges']:.2f}\n"
            )

            file.write(
                f"Churn rate: "
                f"{row['Churn_Rate']:.2f}%\n\n"
            )


# ============================================================
# 28. SAVING SAMPLE CLUSTER ASSIGNMENTS
# ============================================================

sample_assignments = df[
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


print(
    "\n============================================================"
)

print(
    "SAMPLE CLUSTER ASSIGNMENTS"
)

print(
    "============================================================"
)


print(
    sample_assignments
)


sample_assignments.to_csv(
    os.path.join(
        output_folder,
        "Sample_Cluster_Assignments.csv"
    ),
    index=False
)


# ============================================================
# 29. SAVING PCA VISUALIZATION DATA
# ============================================================

pca_df.to_csv(
    os.path.join(
        output_folder,
        "PCA_Cluster_Visualization_Data.csv"
    ),
    index=False
)


# ============================================================
# 30. SAVING CLUSTER SIZE RESULTS
# ============================================================

cluster_sizes_df = (
    cluster_sizes
    .reset_index()
)


cluster_sizes_df.columns = [
    "Cluster",
    "Customer_Count"
]


cluster_sizes_df.to_csv(
    os.path.join(
        output_folder,
        "Cluster_Sizes.csv"
    ),
    index=False
)


# ============================================================
# 31. DISPLAYING GENERATED DELIVERABLES
# ============================================================

print(
    "\n============================================================"
)

print(
    "GENERATED CLUSTERING ANALYSIS DELIVERABLES"
)

print(
    "============================================================"
)


for filename in sorted(
    os.listdir(output_folder)
):

    print(
        " -",
        filename
    )


# ============================================================
# 32. COMPLETING CLUSTERING ANALYSIS
# ============================================================

print(
    "\n============================================================"
)

print(
    "PART 2-2 CLUSTERING ANALYSIS COMPLETED"
)

print(
    "============================================================"
)