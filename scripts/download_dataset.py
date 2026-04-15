"""
Download Telco Customer Churn Dataset
"""
import os
import requests
import pandas as pd

def download_dataset():
    """Download the Telco Customer Churn dataset"""
    
    # Kaggle dataset URL (using a direct download link)
    url = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
    
    output_path = "data/raw/customer_churn.csv"
    
    print("=" * 60)
    print("DOWNLOADING TELCO CUSTOMER CHURN DATASET")
    print("=" * 60)
    print(f"Source: {url}")
    print(f"Destination: {output_path}")
    print()
    
    try:
        # Download the file
        print("Downloading...")
        response = requests.get(url)
        response.raise_for_status()
        
        # Save to file
        os.makedirs("data/raw", exist_ok=True)
        with open(output_path, 'wb') as f:
            f.write(response.content)
        
        # Verify the download
        df = pd.read_csv(output_path)
        print(f"✅ Download successful!")
        print(f"Dataset shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")
        print()
        print("=" * 60)
        print("Dataset ready for training!")
        print("=" * 60)
        
    except Exception as e:
        print(f"❌ Error downloading dataset: {e}")
        print()
        print("Alternative: You can manually download from:")
        print("https://www.kaggle.com/datasets/blastchar/telco-customer-churn")
        print("And place it in: data/raw/customer_churn.csv")

if __name__ == "__main__":
    download_dataset()
