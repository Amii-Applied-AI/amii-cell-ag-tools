import os
import re
import pandas as pd
from dotenv import load_dotenv

load_dotenv(override=False)

"""consistency check between TM Variant and TM with regex-rules"""

INPUT_CSV  = os.environ.get("INPUT_CSV",  "/data/results.csv")
OUTPUT_CSV = os.environ.get("OUTPUT_CSV", "/data/results-flagged-regex.csv")

# If the input is actually an Excel file (or the CSV doesn't exist but an .xlsx does),
# convert it to a temporary CSV first, then read as CSV.
input_path = INPUT_CSV
if input_path.lower().endswith((".xlsx", ".xls")) or (
    not os.path.exists(input_path) and os.path.exists(os.path.splitext(input_path)[0] + ".xlsx")
):
    xlsx_path = input_path if input_path.lower().endswith((".xlsx", ".xls")) else os.path.splitext(input_path)[0] + ".xlsx"
    temp_csv = os.path.splitext(xlsx_path)[0] + ".converted.csv"
    pd.read_excel(xlsx_path, dtype=str).to_csv(temp_csv, index=False)
    input_path = temp_csv

df = pd.read_csv(input_path, dtype=str)

regex_tm = r"([0-9]+(?:\.[0-9]+)?(?:\s*(?:±|\+/-)\s*[0-9]+(?:\.[0-9]+)?)?)\s*°C"

df["Tm_extracted"] = df["TM Variant"].str.extract(regex_tm)
df["Tm_dep_extracted"] = df["Tm"].str.extract(regex_tm, expand=False)

for i in range(len(df["Tm_dep_extracted"])):
    a = df.at[i, "Tm_extracted"]
    b = df.at[i, "Tm_dep_extracted"]
    if pd.isna(a) or pd.isna(b):
        continue
    elif df["Tm_extracted"][i] == df["Tm_dep_extracted"][i]:
        continue
    else:
        print(i, df["Tm_dep_extracted"][i], df["Tm_extracted"][i])

df.to_csv(OUTPUT_CSV, index=False)
