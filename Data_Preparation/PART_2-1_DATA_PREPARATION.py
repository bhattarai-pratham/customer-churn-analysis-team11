import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer


# ============================================================
# PART 2-1: DATA PREPARATION
# ============================================================

# ------------------------------------------------------------
# LOADING DATASET
# ------------------------------------------------------------

file_path = r"D:\project churn\Dataset_ATS_v2.csv"

df = pd.read_csv(file_path)

print("Dataset loaded successfully.")
print("Dataset shape:", df.shape)
print("\nFirst 5 rows:")
print(df.head())


# ------------------------------------------------------------
# 2. CHECKING MISSING VALUES
# ------------------------------------------------------------

print("\nMissing values before preprocessing:")
print(df.isnull().sum())

target_column = "Churn"

# Identifing numerical and categorical columns
numerical_columns = df.select_dtypes(
    include=["int64", "float64"]
).columns.tolist()

categorical_columns = df.select_dtypes(
    include=["object", "category"]
).columns.tolist()

# Removing target from predictor columns
if target_column in categorical_columns:
    categorical_columns.remove(target_column)

# Missing values are handled later, inside the preprocessing
# pipeline (Section 8), fitted on the training set only.
# This avoids leaking test-set statistics into training.


# ------------------------------------------------------------
# 3. REMOVING DUPLICATE RECORDS
# ------------------------------------------------------------

print("\nDuplicate records:", df.duplicated().sum())

df = df.drop_duplicates().reset_index(drop=True)

print("Dataset shape after duplicate removal:", df.shape)


# ------------------------------------------------------------
# 4. SEPARATING FEATURES AND TARGET
# ------------------------------------------------------------

X = df.drop(columns=[target_column])
y = df[target_column]


# ------------------------------------------------------------
# 5. ENCODING TARGET VARIABLE
# ------------------------------------------------------------

y = y.map({
    "No": 0,
    "Yes": 1
})

print("\nTarget distribution:")
print(y.value_counts())


# ------------------------------------------------------------
# 6. IDENTIFING FEATURE TYPES
# ------------------------------------------------------------

numerical_features = X.select_dtypes(
    include=["int64", "float64"]
).columns.tolist()

categorical_features = X.select_dtypes(
    include=["object", "category"]
).columns.tolist()

print("\nNumerical features:")
print(numerical_features)

print("\nCategorical features:")
print(categorical_features)


# ------------------------------------------------------------
# 7. SPLITING DATASET
# ------------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining data shape:", X_train.shape)
print("Testing data shape:", X_test.shape)


# ------------------------------------------------------------
# 8. IMPUTING MISSING VALUES, ENCODING CATEGORICAL VARIABLES,
#    AND SCALING NUMERICAL VARIABLES
#
# Imputation now happens inside the pipeline below, fitted only
# on X_train in Section 9. This prevents test-set statistics
# (medians/modes) from leaking into the training data.
# ------------------------------------------------------------

numerical_pipeline = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_pipeline = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    (
        "encoder",
        OneHotEncoder(
            drop="first",
            handle_unknown="ignore",
            sparse_output=False
        )
    )
])

preprocessor = ColumnTransformer(
    transformers=[
        (
            "numerical",
            numerical_pipeline,
            numerical_features
        ),
        (
            "categorical",
            categorical_pipeline,
            categorical_features
        )
    ]
)


# ------------------------------------------------------------
# 9. FITTING PREPROCESSING ON TRAINING DATA
# ------------------------------------------------------------

X_train_processed = preprocessor.fit_transform(X_train)

X_test_processed = preprocessor.transform(X_test)


# ------------------------------------------------------------
# 10. CONVERTING PROCESSED DATA TO DATAFRAMES
# ------------------------------------------------------------

feature_names = preprocessor.get_feature_names_out()

X_train_processed = pd.DataFrame(
    X_train_processed,
    columns=feature_names
)

X_test_processed = pd.DataFrame(
    X_test_processed,
    columns=feature_names
)


# ------------------------------------------------------------
# 11. PREPROCESSED DATA VISUALIZATION
# ------------------------------------------------------------

print("\nProcessed Training Data:")
print(X_train_processed.head())

print("\nProcessed Testing Data:")
print(X_test_processed.head())

print("\nNumber of processed features:",
      X_train_processed.shape[1])


# ------------------------------------------------------------
# 12. SUMMARY
# ------------------------------------------------------------

print("Training samples:", len(X_train_processed))
print("Testing samples:", len(X_test_processed))
print("Processed features:", X_train_processed.shape[1])

print("\nTraining target distribution:")
print(y_train.value_counts())

print("\nTesting target distribution:")
print(y_test.value_counts())

# ============================================================
# 13. SAVE PREPROCESSED DATASETS
# ============================================================

import os

output_folder = r"D:\project churn\Part_1_Output"

os.makedirs(output_folder, exist_ok=True)


# ------------------------------------------------------------
# Save cleaned dataset
# ------------------------------------------------------------

cleaned_dataset = df.copy()

cleaned_dataset.to_csv(
    os.path.join(
        output_folder,
        "Cleaned_Dataset.csv"
    ),
    index=False
)


# ------------------------------------------------------------
# Save preprocessed training dataset
# ------------------------------------------------------------

preprocessed_training = X_train_processed.copy()

preprocessed_training["Churn"] = y_train.values

preprocessed_training.to_csv(
    os.path.join(
        output_folder,
        "Preprocessed_Training_Dataset.csv"
    ),
    index=False
)


# ------------------------------------------------------------
# Save preprocessed testing dataset
# ------------------------------------------------------------

preprocessed_testing = X_test_processed.copy()

preprocessed_testing["Churn"] = y_test.values

preprocessed_testing.to_csv(
    os.path.join(
        output_folder,
        "Preprocessed_Testing_Dataset.csv"
    ),
    index=False
)


# ------------------------------------------------------------
# Saving complete preprocessed dataset
# ------------------------------------------------------------

preprocessed_dataset = pd.concat(
    [
        preprocessed_training,
        preprocessed_testing
    ],
    ignore_index=True
)

preprocessed_dataset.to_csv(
    os.path.join(
        output_folder,
        "Preprocessed_Dataset.csv"
    ),
    index=False
)


print("\n============================================================")
print("OUTPUT FILES CREATED")
print("============================================================")

print(
    "Cleaned Dataset:",
    os.path.join(
        output_folder,
        "Cleaned_Dataset.csv"
    )
)

print(
    "Preprocessed Dataset:",
    os.path.join(
        output_folder,
        "Preprocessed_Dataset.csv"
    )
)

print(
    "Training Dataset:",
    os.path.join(
        output_folder,
        "Preprocessed_Training_Dataset.csv"
    )
)

print(
    "Testing Dataset:",
    os.path.join(
        output_folder,
        "Preprocessed_Testing_Dataset.csv"
    )
)
