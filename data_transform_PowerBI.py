import argparse
import pandas as pd 
import numpy as np
import wfdb
from tqdm import tqdm

def main():
    
    parser = argparse.ArgumentParser(description="Download ECG signals from PhysioNet for Power BI")
    parser.add_argument('-r', '--record', type=str, required=True, help="ID record to download (np. '100', '205')")
    parser.add_argument('-d', '--db', type=str, default='mitdb', help="Name of the database (default: mitdb)")
    
    args = parser.parse_args()
    record_id = args.record
    db_name = args.db

    print(f"Connecting to PhysioNet: Attempting to download record '{record_id}' from database '{db_name}'...")

    window_size = 1024

    try:
        
        signal, fields = wfdb.rdsamp(record_id, pn_dir=db_name, channels=[0])
        signal = signal.flatten()
        
        
        num_windows = len(signal) // window_size
        signal_trimmed = signal[:num_windows * window_size]
        windows = signal_trimmed.reshape((num_windows, window_size))
        
        
        windows_rounded = np.round(windows, 4)
        signal_strings = [",".join(map(str, window)) for window in windows_rounded]
        
        
        df = pd.DataFrame({
            'Record_Id': record_id,
            'Okno': np.arange(num_windows),
            'Sygnal': signal_strings
        })
        
    
        output_filename = f'physionet_{db_name}_record_{record_id}.csv'
        df.to_csv(output_filename, index=False)
        
        print(f"✅ Success! Generated file: {output_filename}")
        print(f"Extracted {num_windows} windows (of {window_size} samples) for record {record_id}.")
        
    except ValueError as ve:
        print(f"❌ Error: Record '{record_id}' not found in database '{db_name}'. Please ensure the ID is correct.")
    except Exception as e:
        print(f"❌ An unexpected error occurred: {e}")

if __name__ == "__main__":
    main()