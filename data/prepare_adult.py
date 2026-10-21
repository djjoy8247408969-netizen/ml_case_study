"""
Builds data/adult.csv from the two raw UCI Adult files (adult.data + adult.test).

Raw-file quirks handled here:
  * no header row  -> column names added from adult.names
  * values padded with a leading space (", Private") -> stripped
  * adult.test starts with a junk line ("|1x3 Cross validator") -> skipped
  * adult.test labels end with a full stop ("<=50K.") -> stripped so both files share the labels '<=50K' / '>50K'
  * missing values are left as the literal '?' -> the notebook parses them with na_values=['?']

Usage (from the repo root):  python data/prepare_adult.py
"""
from pathlib import Path
import pandas as pd

HERE = Path(__file__).resolve().parent
RAW = HERE / 'raw_adult'
COLUMNS = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status',
           'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss',
           'hours-per-week', 'native-country', 'income']

train = pd.read_csv(RAW / 'adult.data', header=None, names=COLUMNS, skipinitialspace=True)
test = pd.read_csv(RAW / 'adult.test', header=None, names=COLUMNS, skipinitialspace=True, skiprows=1)
test['income'] = test['income'].str.rstrip('.')

adult = pd.concat([train, test], ignore_index=True)
adult.to_csv(HERE / 'adult.csv', index=False)
print(f"adult.csv written: {adult.shape[0]:,} rows x {adult.shape[1]} columns")
print(adult['income'].value_counts())
