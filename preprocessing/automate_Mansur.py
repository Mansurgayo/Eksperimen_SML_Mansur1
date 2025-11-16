# UPDATE FINAL: SCALER SUDAH DIKIRIM
import pandas as pd
from sklearn.preprocessing import StandardScaler
import os

RAW_DATA_PATH = 'namadataset_raw/diabetes.csv'


PROCESSED_DATA_FOLDER = 'namadataset_preprocessing' 


PROCESSED_DATA_PATH = os.path.join(PROCESSED_DATA_FOLDER, 'cleaned_dataset.csv')
# -----------------------------

def load_data(raw_path):
    """Memuat data dari path file mentah."""
    print(f"Memuat data dari: {raw_path}")
    df = pd.read_csv(raw_path)
    print("Data berhasil dimuat.")
    return df

def preprocess_data(df):
    """Membersihkan dan memproses data."""
    print("Memulai preprocessing...")
    df_processed = df.copy()
    
    cols_with_zero_issue = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
    
  
    for col in cols_with_zero_issue:
        df_processed[col] = df_processed[col].replace(0, pd.NA)
        
  
    for col in cols_with_zero_issue:
        if df_processed[col].isnull().any():
            median_val = df_processed[col].median()
            df_processed[col] = df_processed[col].fillna(median_val)
            print(f"Mengisi NaN di '{col}' dengan median: {median_val}")
    
    if 'Outcome' not in df_processed.columns:
        print("Error: Kolom 'Outcome' tidak ditemukan.")
        return None
        
    X = df_processed.drop('Outcome', axis=1)
    y = df_processed['Outcome']
    
    print("Melakukan feature scaling (StandardScaler)...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    df_cleaned = pd.DataFrame(X_scaled, columns=X.columns)
    df_cleaned['Outcome'] = y.values
    print("Preprocessing selesai.")
    return df_cleaned

def save_data(df, processed_path):
    """Menyimpan data yang sudah bersih ke path tujuan."""
    if df is None:
        print("Tidak ada data untuk disimpan.")
        return


    os.makedirs(os.path.dirname(processed_path), exist_ok=True)
    
    
    df.to_csv(processed_path, index=False)
    print(f"Data bersih BERHASIL DISIMPAN di: {processed_path}")

if __name__ == '__main__':
    print("Menjalankan pipeline preprocessing otomatis...")
    df = load_data(RAW_DATA_PATH)
    df_cleaned = preprocess_data(df)
    save_data(df_cleaned, PROCESSED_DATA_PATH)
    print("Pipeline selesai.")