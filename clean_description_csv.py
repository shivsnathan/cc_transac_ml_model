import pandas as pd
import re

# Load your DataFrame
df = pd.read_csv("parsed_transactions.csv")

# Regex to remove leading date-like patterns at the beginning of the description
df['Description'] = df['Description'].str.replace(
    r'^(?:\d{2}[/-]\d{2}(?:[/-]\d{2,4})?|\d{2}\s?[A-Za-z]{3}(?:\s?\d{2,4})?)\s+', '', 
    regex=True
)

# Save cleaned file
df.to_csv("parsed_transactions_cleaned.csv", index=False)
print("✅ Cleaned all date prefixes from Description column.")
