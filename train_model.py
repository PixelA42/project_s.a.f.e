import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv('labels.csv')

#prevent data leakage
# Get a list of the unique original audio files
unique_files = df['original_audio'].unique()
#splittting data 60-40
train_files, test_files = train_test_split(unique_files, test_size=0.4, random_state=42)

# Create the final DataFrames based on the split files
train_df = df[df['original_audio'].isin(train_files)]
test_df = df[df['original_audio'].isin(test_files)]

print("=" * 50)
print(" DATASET SPLIT COMPLETE")
print("=" * 50)
print(f" Total Images    : {len(df)}")
print(f" Training Images : {len(train_df)} (60%)")
print(f" Testing Images  : {len(test_df)} (40%)")

# ==========================================
# 3. SEPARATE FEATURES (X) AND LABELS (y)
# ==========================================
X_train = train_df['file_path'].values
y_train = train_df['label'].values

X_test = test_df['file_path'].values
y_test = test_df['label'].values

