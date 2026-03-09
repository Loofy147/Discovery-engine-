import re
import pandas as pd
df = pd.read_csv('reference.csv')
p = df.iloc[8]['problem']
print(f"P9 Raw: {repr(p)}")
raw_norm = re.sub(r'[^a-z0-9]', '', p.lower())
print(f"P9 Norm: {raw_norm[:100]}")
