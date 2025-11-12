# Growth factor thermostability dataset to accelerate cultured meat and seafood development

We provide here the source code and scripts used to generate the publicly available dataset located here: 

* Link to Dataset ... ``` Link to paper ```

This repository supports the paper detailing the creation of a high-quality, open-source dataset on **Growth Factor (GF) thermostability** to accelerate the development and regulatory acceptance of **Cultured Meat and Seafood (CM)**.

The paper addresses the critical need for public safety data on GFs—essential CM ingredients—by focusing on **melting temperature (T<sub>m</sub>)**, the most direct measure of protein stability. The project combines two data generation methods:

1. AI-Guided Curation: Automated extraction and rigorous manual validation of existing T<sub>m</sub> data from scientific literature.  (The focus of this library)

2. In Silico Analysis: Generation of thermodynamic protein features (like folding free energy ($\Delta T$)) using FoldX software. (Data that is later merged in with the data from #1)

The resulting dataset provides the essential foundation for safety evaluation, product development, and the creation of AI/ML tools to predict and engineer safer, more effective growth factors for the CM industry.

## Flow

1- Run Column_extranction-DeepSeek.py → first version (one row per TM variant)

2- Do manual checks → make manual_tms.csv

3- Run Column_extranction-GPT → refined version that uses your reviewed TMs

4- Validation ( run consistency-check.py and consistency-check-regex-rules.py to get flagged for inconsistencies )

## Folder Layout

Files in papers/ are grouped by the prefix before the first - or _:
```
papers/
  Smith 2019 - Title.pdf
  Smith 2019 - Supplement.pdf
  Lee_2021 - Paper.pdf
  Lee_2021 - SI.pdf
```
Group keys here are Smith 2019 and Lee_2021.

## 📦 Installation

### System Requirements
- Python **3.9–3.12**
- [Tesseract OCR](https://tesseract-ocr.github.io/)  
  - macOS:  
    ```bash
    brew install tesseract
    ```
  - Ubuntu/Debian:  
    ```bash
    sudo apt-get update && sudo apt-get install -y tesseract-ocr
    ```

### Python Dependencies

Install using the provided `requirements.txt`:

```requirements.txt``` installs the DeepSeek-compatible OpenAI SDK (v1) by default.

```
python -m venv .venv-deepseek
source .venv-deepseek/bin/activate   # Windows: .venv-deepseek\Scripts\activate
pip install -r requirements.txt
```

If you want to run the GPT scripts, edit requirements.txt:

- Comment out the whole deepseek section including  ```openai>=1.12.0,<2.0``` line
- Uncomment the GPT section (including ```openai==0.28.1```)
- Then create a fresh virtualenv and reinstall:
```
deactivate      # deactivate the deepseek venv
python -m venv .venv-gpt
source .venv-gpt/bin/activate      # Windows: .venv-gpt\Scripts\activate
pip install -r requirements.txt
```
If you hit client-version conflicts (DeepSeek uses OpenAI client ≥1.0; GPT script uses the legacy openai.ChatCompletion), use two venvs.

## 🔑 Setup
```
Create a .env file (you can copy example.env):

# Common paths
PDF_FOLDER=/abs/path/to/protein-thermostability/papers
TM_CSV=/abs/path/to/protein-thermostability/manual_tms.csv

# DeepSeek 
DEEPSEEK_API_KEY=""
DEEPSEEK_BASE_URL=https://api.deepseek.com
DEEPSEEK_MODEL=deepseek-reasoner
DEEPSEEK_OUTPUT=/abs/path/to/papers-deepseek.xlsx

# GPT 
OPENAI_API_KEY=""
GPT_OUTPUT=/abs/path/to/papers-gpt.xlsx
GPT_MODEL=gpt-4.1

# Consistency / Regex checks
INPUT_CSV=/abs/path/to/results.csv
OUTPUT_CSV=/abs/path/to/flagged-results.csv   # (for judge)
JUDGE_MODEL=o4-mini
CALL_DELAY_SEC=0
```

then run this code:
```cp example.env .env```

You can override these values with CLI flags.

## 🚀 Usage

1) DeepSeek pass 

Runs over each group, detects TM variants, and writes one row per TM variant.

```
export OUTPUT_EXCEL="${DEEPSEEK_OUTPUT}" # optional: If you want to override the example.env file
```

```
source .venv-deepseek/bin/activate
python column_extraction-deepseek.py
# writes $DEEPSEEK_OUTPUT
```
Output: ${DEEPSEEK_OUTPUT} (Excel).

2) Manual checks (build your TM list)

Create manual_tms.csv at the path in TM_CSV with these additional columns:

| Key        | TM Variant       | Correct or not? | Revised or Not.       |
| :--------- | :--------------- | :-------------- | :-------------------- |
| Smith 2019 | EGF WT           | correct         |                       |
| Smith 2019 | EGF L52I         | revised         | EGF L52I (stabilized) |
| Lee_2021   | HGF NK1          | added           | HGF NK1 (mutant)      |

**Rules used by the GPT script:**
- If `Correct or not?` contains **correct** → keep **TM Variant**.
- If it contains **revised** or **added** → use **Revised or Not.**.
- Multiple rows per **Key** are allowed (each becomes a separate row in the final sheet).
- **Key** must match the file-group prefix exactly (e.g., `Smith 2019`).


3) GPT pass (refined, uses your manual TMs)
   
```
export OUTPUT_EXCEL="${GPT_OUTPUT}" # optional: If you want to override the example.env file
```

```
source .venv-gpt/bin/activate
python column_extraction-gpt.py
# writes $GPT_OUTPUT
```

Output: ${GPT_OUTPUT} (Excel).

4) Validation

A) Tm regex sanity:

  Pulls a numeric Tm from TM Variant text and compares it to the Tm column.
  
```
# Optional: If you want to override example.env
export INPUT_CSV="$GPT_OUTPUT"
export OUTPUT_CSV="./flags-tm-regex.csv"
```

```
python consistency-check-regex-rules.py
# check console mismatches or filter rows where Tm_extracted ≠ Tm_dep_extracted
```

B) Using LLM as a judge for consistency check:

  Reads ```Experimental Conditions/Methods Description``` and verifies the structured fields ```Buffer```, ```pH```, ```Protein concentration``` are actually stated and match.
  
  Outputs: ```buffer_ok```, ```pH_ok```, ```conc_ok```, plus ```inconsistent_fields``` and ```any_inconsistent```.
  
  Example: description “50 mM Tris, pH 7.5” but row has “PBS, pH 7.4” → ```buffer_ok=false```, ```pH_ok=false```.
  
```
# Optional: If you want to override example.env
export INPUT_CSV="$GPT_OUTPUT"
export OUTPUT_CSV="./flags-judge.csv"
```

```
python consistency-check.py
# focus on rows with any_inconsistent == true
```

## Example

**DeepSeek output** ($DEEPSEEK_OUTPUT)
| Author/Year of Publication | Paper Title                                   | Species Origin                                                                                            | Production Organism                                                                                                                                                                                                             | TM Variant  | Growth Factor | Amino Acid Sequence                                                                                                                                                                                                                           | Methods                                                                                                                                   | Experimental Conditions/Methods Description                                                                                                                                                                                                                                                   | pH  | Buffer                                                                                         | Protein concentration                                                                                                                     | Tm                                                                                                         | Engineered variant? | PDF File Name                                                    |
| :------------------------- | :-------------------------------------------- | :-------------------------------------------------------------------------------------------------------- | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | :---------- | :------------ | :-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | :---------------------------------------------------------------------------------------------------------------------------------------- | :-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | :-- | :--------------------------------------------------------------------------------------------- | :---------------------------------------------------------------------------------------------------------------------------------------- | :--------------------------------------------------------------------------------------------------------- | :------------------ | :--------------------------------------------------------------- |
| Vinther et al., 2013       | Insulin analog with additional disulfide bond | Human (Homo sapiens) insulin was used; expression and study were based on recombinant production systems. | Saccharomyces cerevisiae  Source: Section "Materials and Methods," particularly in the description of "Insulin-like single-chains..." (Materials and Methods, Plasmids construction) where yeast-based expression is described. | HI          | Insulin       | UniProt ID (wild-type insulin): P01308  Mutations: Not applicable for HI (wild-type insulin).                                                                                                                                                 | Differential Scanning Calorimetry (DSC)  Source: "Differential scanning calorimetry (DSC) measurements"; "Materials and Methods" section. | The DSC measurements were performed using an insulin concentration of 0.2 mM in phosphate buffer (pH 7.5), with a scan rate of 1 °C/min. Additional conditions: temperature range appropriately set for insulin thermal transitions; the buffer composition maintained at low ionic strength. | 7.5 | 0.2 mM phosphate buffer pH 7.5  Source: “Differential scanning calorimetry (DSC) measurements” | 0.2 mM  Source: "Materials and Methods", section "Differential scanning calorimetry (DSC) measurements" indicating insulin concentration. | 64.2 °C  Source: Main text, Figure 5A and corresponding text discussing the thermal stability (Tm) for HI. | No.                 | Vinther 2013 - Insulin analog with additional disulfide bond.pdf |
| Vinther et al., 2013       | Insulin analog with additional disulfide bond | Human (Homo sapiens) insulin was used; expression and study were based on recombinant production systems. | Saccharomyces cerevisiae  Source: Section "Materials and Methods," particularly in the description of "Insulin-like single-chains..." (Materials and Methods, Plasmids construction) where yeast-based expression is described. | 4SS-insulin | Insulin       | UniProt ID (wild-type insulin): P01308  Mutations: Engineered variant featuring an additional disulfide bond; specific cysteine substitutions that create the fourth disulfide; location noted in main text describing “4-disulfide insulin”. | Differential Scanning Calorimetry (DSC)  Source: "Differential scanning calorimetry (DSC) measurements"; "Materials and Methods" section. | DSC conditions: insulin concentration 0.2 mM in phosphate buffer (pH 7.5); scan rate of 1 °C/min; measurements reflect enhanced stability of the engineered variant with additional disulfide linkage.                                                                                        | 7.5 | 0.2 mM phosphate buffer pH 7.5  Source: Main text and “Materials and Methods,” DSC section     | 0.2 mM  Source: "Differential scanning calorimetry (DSC) measurements" section                                                            | 98.8 °C  Source: Main text, Figure 5A and the discussion comparing HI and 4SS-insulin Tm values            | Yes.                | Vinther 2013 - Insulin analog with additional disulfide bond.pdf |


**Manual checks** ($TM_CSV)


| Author/Year of Publication | Paper Title                                   | Species Origin | Production Organism      | TM Variant            | Revised or Not. | Correct or not? | PDF File Name                                                    | Key          |
| :------------------------- | :-------------------------------------------- | :------------- | :----------------------- | :-------------------- | :-------------- | :-------------- | :--------------------------------------------------------------- | :----------- |
| Vinther et al., 2013       | Insulin analog with additional disulfide bond | Human insulin  | Saccharomyces cerevisiae | HI (64.2 °C)          |                 | Correct         | Vinther 2013 - Insulin analog with additional disulfide bond.pdf | Vinther 2013 |
| Vinther et al., 2013       | Insulin analog with additional disulfide bond | Human insulin  | Saccharomyces cerevisiae | 4SS-insulin (98.8 °C) |                 | Correct         | Vinther 2013 - Insulin analog with additional disulfide bond.pdf | Vinther 2013 |




**GPT output** ($GPT_OUTPUT)

| Author/Year of Publication | Paper Title                                   | Species Origin                                                                                        | Production Organism                                                                                                              | TM Variant  | Growth Factor | Amino Acid Sequence                                                                                                  | Methods                                                                                                           | Experimental Conditions/Methods Description                                                                                                | pH  | Buffer                                                        | Protein concentration                   | Tm                                                                         | Engineered variant? | PDF File Name                                                    |
| :------------------------- | :-------------------------------------------- | :---------------------------------------------------------------------------------------------------- | :------------------------------------------------------------------------------------------------------------------------------- | :---------- | :------------ | :------------------------------------------------------------------------------------------------------------------- | :---------------------------------------------------------------------------------------------------------------- | :----------------------------------------------------------------------------------------------------------------------------------------- | :-- | :------------------------------------------------------------ | :-------------------------------------- | :------------------------------------------------------------------------- | :------------------ | :--------------------------------------------------------------- |
| Vinther et al., 2013       | Insulin analog with additional disulfide bond | Human insulin (HI) is the species origin related to human; the paper discusses human insulin analogs. | Saccharomyces cerevisiae  Source: "The insulin precursors were expressed in Saccharomyces cerevisiae..." (Materials and Methods) | HI          | Insulin       | UniProt ID: P01308  Mutations: none (wild-type insulin)                                                              | Differential Scanning Calorimetry (DSC)  Source: “Differential scanning calorimetry (DSC) measurements” (Methods) | 0.2 mM insulin in phosphate buffer at pH 7.5; DSC scan rate 1 °C/min; temperature program as described for insulin stability measurements. | 7.5 | Buffer: 0.2 mM phosphate buffer pH 7.5  Source: Methods / DSC | 0.2 mM  Source: Methods / DSC           | 64.2 °C  Source: Figure 5 and text, main paper                             | No.                 | Vinther 2013 - Insulin analog with additional disulfide bond.pdf |
| Vinther et al., 2013       | Insulin analog with additional disulfide bond | Human insulin (HI) is the species origin related to human; the paper discusses human insulin analogs. | Saccharomyces cerevisiae  Source: "The insulin precursors were expressed in Saccharomyces cerevisiae..." (Materials and Methods) | 4SS-insulin | Insulin       | UniProt ID: P01308  Mutations: engineered addition of a fourth disulfide bond; cysteine positions described in text. | Differential Scanning Calorimetry (DSC)  Source: “Differential scanning calorimetry (DSC) measurements” (Methods) | Conditions matched to HI: 0.2 mM protein, phosphate buffer pH 7.5; DSC scan 1 °C/min; measured higher stability vs. HI.                    | 7.5 | 0.2 mM phosphate buffer, pH 7.5  Source: Main text / Methods  | 0.2 mM  Source: “Materials and Methods” | 98.8 °C  Source: Main text, Figure 5A and corresponding comparison with HI | Yes.                | Vinther 2013 - Insulin analog with additional disulfide bond.pdf |



**Validation Flags**

| Author/Year of Publication | Paper Title                                   | Species Origin                                                                                        | Production Organism                                                                                                              | TM Variant  | Growth Factor | Amino Acid Sequence                                                                                                  | Methods                                 | Experimental Conditions/Methods Description                                         | pH  | Buffer                  | Protein concentration | Tm      | Engineered variant? | PDF File Name                                                    | buffer_ok | pH_ok | conc_ok | inconsistent_fields | any_inconsistent |
| :------------------------- | :-------------------------------------------- | :---------------------------------------------------------------------------------------------------- | :------------------------------------------------------------------------------------------------------------------------------- | :---------- | :------------ | :------------------------------------------------------------------------------------------------------------------- | :-------------------------------------- | :---------------------------------------------------------------------------------- | :-- | :---------------------- | :-------------------- | :------ | :------------------ | :--------------------------------------------------------------- | :-------- | :---- | :------ | :------------------ | :--------------- |
| Vinther et al., 2013       | Insulin analog with additional disulfide bond | Human insulin (HI) is the species origin related to human; the paper discusses human insulin analogs. | Saccharomyces cerevisiae  Source: "The insulin precursors were expressed in Saccharomyces cerevisiae..." (Materials and Methods) | HI          | Insulin       | UniProt ID: P01308  Mutations: none (wild-type insulin)                                                              | Differential Scanning Calorimetry (DSC) | DSC conditions for HI; source lines specify “phosphate buffer pH 7.5” and 0.2 mM    | 7.5 | phosphate buffer pH 7.5 | 0.2 mM                | 64.2 °C | No.                 | Vinther 2013 - Insulin analog with additional disulfide bond.pdf | True      | True  | True    |                     | False            |
| Vinther et al., 2013       | Insulin analog with additional disulfide bond | Human insulin (HI) is the species origin related to human; the paper discusses human insulin analogs. | Saccharomyces cerevisiae  Source: "The insulin precursors were expressed in Saccharomyces cerevisiae..." (Materials and Methods) | 4SS-insulin | Insulin       | UniProt ID: P01308  Mutations: engineered addition of a fourth disulfide bond; cysteine positions described in text. | Differential Scanning Calorimetry (DSC) | DSC conditions for engineered 4SS-insulin; matched buffer and concentration with HI | 7.5 | phosphate buffer pH 7.5 | 0.2 mM                | 98.8 °C | Yes.                | Vinther 2013 - Insulin analog with additional disulfide bond.pdf | True      | True  | True    |                     | False            |



## ❗ Notes & Troubleshooting
```
API keys: Set via .env or --api-key. If you hit rate limits, add --delay 0.2.

OCR issues: Ensure Tesseract is installed and images are at least 300 DPI.

Scanned PDFs: Require OCR (pytesseract).

OpenAI/DeepSeek: Make sure you uncomment the correct openai version in requirements.txt.
```


## Contribution/Credit

The library was created and is maintained by [@kianak2002](https://github.com/kianak2002). and [@staszakAmii](https://github.com/staszakAmii).

Contributions and suggestions are welcome! Feel free to download, use, create issues or submit pull requests.

## License

This project is licensed under the [MIT License](LICENSE).
