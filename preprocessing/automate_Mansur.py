import pandas as pd
import numpy as np
import argparse
import os

def preprocess_data(input_path, output_path):
    """
    Memuat data mentah dari input_path, melakukan preprocessing
    (sesuai notebook Anda), dan menyimpannya ke output_path.
    """
    print(f"Memulai preprocessing untuk {input_path}...")
    
    try:
        df = pd.read_csv(input_path)
        print("Data mentah berhasil dimuat.")
    except FileNotFoundError:
        print(f"Error: File tidak ditemukan di {input_path}")
        return

    
    
    
    cols_with_zero = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
    for col in cols_with_zero:
        df[col] = df[col].replace(0, np.nan)
    print("Nilai 0 diganti dengan NaN.")

    
    df.fillna(df.median(), inplace=True)
    print("Nilai NaN diisi dengan median.")

    
    df.drop_duplicates(inplace=True)
    print("Data duplikat dihapus.")
    
    
    numeric_cols = df.select_dtypes(include=np.number).columns.drop('Outcome')
    Q1 = df[numeric_cols].quantile(0.25)
    Q3 = df[numeric_cols].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    condition = ((df[numeric_cols] < lower_bound) | (df[numeric_cols] > upper_bound)).any(axis=1)
    df_cleaned = df[~condition]
    print(f"Data setelah outlier removal: {df_cleaned.shape}")
    
    
    
    
    output_dir = os.path.dirname(output_path)
    
    
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 6. Simpan Data yang Sudah Diproses
    df_cleaned.to_csv(output_path, index=False)
    print(f"Preprocessing selesai. Data bersih disimpan di {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script otomatisasi preprocessing data diabetes.")
    

    parser.add_argument(
        '--input', 
        type=str, 
        default='../namadataset_raw/diabetes.csv', 
    )
    
    parser.add_argument(
        '--output', 
        type=str, 
        default='cleaned_dataset.csv', 
        help='Path untuk menyimpan file hasil preprocessing'
    )
    
    args = parser.parse_args()
    
    preprocess_data(input_path=args.input, output_path=args.output)