import pandas as pd

def load_data(file):
    df = pd.read_csv(file)

    df.columns = df.columns.str.strip()

    # Convert potential date columns to datetime
    for col in df.columns:
        try:
            df[col] = pd.to_datetime(df[col])
        except Exception:
            pass  # Ignore non-date columns

    # Convert numeric columns (force conversion)
    for col in df.columns:
        try:
            df[col] = pd.to_numeric(df[col], errors='coerce')  # Converts numbers stored as strings
        except Exception:
            pass

    # Drop fully empty columns
    df.dropna(axis=1, how='all', inplace=True)

    return df
