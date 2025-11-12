import os
import time
import json
import pandas as pd
import openai
from dotenv import load_dotenv
from openai.error import RateLimitError, APIError, APIConnectionError

# --- Configuration ---
load_dotenv(override=False)

openai.api_key   = os.environ.get("OPENAI_API_KEY", "")  # api key
INPUT_CSV        = os.environ.get("INPUT_CSV") or "/data/result.csv"
OUTPUT_CSV       = os.environ.get("OUTPUT_CSV") or "/data/flagged-result.csv"
CALL_DELAY_SEC   = float(os.environ.get("CALL_DELAY_SEC", "0"))  # pause between LLM calls
MODEL_NAME       = os.environ.get("JUDGE_MODEL", "o4-mini")

if not openai.api_key:
    raise RuntimeError("OPENAI_API_KEY not set. Put it in your environment or in a .env file.")

input_path = INPUT_CSV
if input_path.lower().endswith((".xlsx", ".xls")) or (
    not os.path.exists(input_path) and os.path.exists(os.path.splitext(input_path)[0] + ".xlsx")
):
    xlsx_path = input_path if input_path.lower().endswith((".xlsx", ".xls")) else os.path.splitext(input_path)[0] + ".xlsx"
    temp_csv = os.path.splitext(xlsx_path)[0] + ".converted.csv"
    pd.read_excel(xlsx_path, dtype=str, engine="openpyxl").to_csv(
        temp_csv, index=False, encoding="utf-8-sig", lineterminator="\n"
    )
    input_path = temp_csv

df = pd.read_csv(input_path, dtype=str).fillna("")

"""LLM as a Judge
Gets the Eperimental condition, Buffer, PH, Protein Concentration columns
check if the info are consistent with each other
"""

def judge_consistency(desc, buff, ph_val, conc):
    system_msg = {
        "role": "system",
        "content": "You are an expert data auditor. You compare free-text method descriptions against extracted values."
    }
    user_msg = {
        "role": "user",
        "content": f"""
We have one paper’s data in four columns:
1) Experimental Conditions/Methods Description (a free-text blob),
2) Buffer,
3) pH,
4) Protein concentration.

For **each** of the three structured values (Buffer, pH, Protein concentration):
- If the free-text **mentions** that exact value, respond whether it’s **correct** (YES) or **incorrect** (NO).
- If the free-text does **not mention** that value at all, respond **null**.  

Here are the values:

Experimental Conditions/Methods Description:
\"\"\"{desc[:2000]}\"\"\"

Structured values:
- Buffer: \"{buff}\"
- pH: \"{ph_val}\"
- Protein concentration: \"{conc}\"

**Output valid JSON** with exactly these keys mapping to `true` / `false` / `null`:

For each value, answer YES if the description clearly supports it, NO otherwise.
Output valid JSON with keys "buffer", "pH", "concentration" mapping to true/false.
Example:
{{"buffer": true, "pH": false, "concentration": null}}
        """.strip()
    }
    backoff = 1
    for _ in range(3):
        try:
            resp = openai.ChatCompletion.create(
                model=MODEL_NAME,
                messages=[system_msg, user_msg]
            )
            return json.loads(resp.choices[0].message.content)
        except (RateLimitError, APIError, APIConnectionError):
            time.sleep(backoff)
            backoff = min(backoff*2, 60)
    # if still failing, mark all false
    return {"buffer": False, "pH": False, "concentration": False}

""" Create 'inconsistent_fields' summary """
def summarize_inconsistency(r):
    bad = []
    if not r["buffer_ok"]:
        bad.append("Buffer")
    if not r["pH_ok"]:
        bad.append("pH")
    if not r["conc_ok"]:
        bad.append("Protein concentration")
    return "; ".join(bad)

flags = []
for _, row in df.iterrows():
    desc   = row.get("Experimental Conditions/Methods Description", "")
    buff   = row.get("Buffer", "")
    ph_val = row.get("pH", "")
    conc   = row.get("Protein concentration", "")
    verdict = judge_consistency(desc, buff, ph_val, conc)
    flags.append(verdict)
    time.sleep(CALL_DELAY_SEC)

flags_df = pd.DataFrame(flags)
flags_df.columns = ["buffer_ok", "pH_ok", "conc_ok"]
df = pd.concat([df, flags_df], axis=1)

df["inconsistent_fields"] = df.apply(summarize_inconsistency, axis=1)
df["any_inconsistent"]    = df["inconsistent_fields"] != ""

df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig", lineterminator="\n")
print(f"✅ Consistency flags added and saved to {OUTPUT_CSV}")
