import pdfplumber
import pandas as pd
import os
from dateutil import parser
import re

input_folder = "/Users/siva/Downloads/credit card statement/"
output_csv = "parsed_transactions.csv"

def parse_date_safe(date_str):
    try:
        return parser.parse(date_str, dayfirst=True).date()
    except:
        return None

all_data = []

for file in os.listdir(input_folder):
    if not file.endswith(".pdf"):
        continue

    file_path = os.path.join(input_folder, file)
    
    # Infer bank name from filename (take first word before _ or space)
    base_name = os.path.basename(file).split('.')[0]
    bank_name = re.split(r"[ _]", base_name)[0]  # e.g., 'HDFC_Card_March' → 'HDFC'

    with pdfplumber.open(file_path) as pdf:
        for page in pdf.pages:
            words = page.extract_words()

            line_buffer = []
            last_top = None
            for word in words:
                top = round(word['top'], 1)
                if last_top is None or abs(top - last_top) < 2:
                    line_buffer.append(word)
                else:
                    line_text = " ".join(w['text'] for w in line_buffer)
                    parts = line_text.split()
                    if len(parts) >= 3:
                        try:
                            txn_date = parse_date_safe(parts[0])
                            amount = float(parts[-1].replace(",", "").replace("AED", "").strip())
                            description = " ".join(parts[1:-1])
                            if txn_date:
                                all_data.append([txn_date, description, amount, bank_name, ""])
                        except:
                            pass
                    line_buffer = [word]
                last_top = top

            # Process final buffer
            if line_buffer:
                line_text = " ".join(w['text'] for w in line_buffer)
                parts = line_text.split()
                if len(parts) >= 3:
                    try:
                        txn_date = parse_date_safe(parts[0])
                        amount = float(parts[-1].replace(",", "").replace("AED", "").strip())
                        description = " ".join(parts[1:-1])
                        if txn_date:
                            all_data.append([txn_date, description, amount, bank_name, ""])
                    except:
                        pass

# Create DataFrame with enhanced columns
df = pd.DataFrame(all_data, columns=["Date", "Description", "Amount", "Bank", "Category"])
df.to_csv(output_csv, index=False)
print(f"✅ Enhanced CSV saved with {len(df)} transactions to {output_csv}")
