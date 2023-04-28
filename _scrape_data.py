import yfinance as yf
from datetime import date
from datetime import timedelta
import os

print(f"Running ... {os.path.basename(__file__)}\n")

if __name__ == "__main__":
    stock = input("Stock : ")
    year = input("Start Date Year : ")
    month = input("Start Date Month : ")
    day = input("Start Date Day : ")

    print("\n\nExtract Data:\n")
    dir_path = os.path.dirname(__file__)
    print(f"Downloaded data saved to same folder as script.py: \n{dir_path}")

    # Date Range
    start = f'{year}-{month}-{day}'
    yesterday = date.today() - timedelta(days=1)
    end = yesterday.strftime("%Y-%m-%d")
    print(f"\nExtract Date from {start} to {end}")

    # Download
    data = yf.download(stock, start, end)

    # Create Folder if don't exist
    if not os.path.exists("dataset"):
        os.mkdir("dataset")

    # Save to csv file
    data.to_csv(dir_path +"\\"+ "dataset\data.csv")
    input("\n\nEnter to close")
