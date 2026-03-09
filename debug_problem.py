import pandas as pd
import re
df = pd.read_csv('reference.csv')
p1 = df.iloc[0]['problem']
print(f"P1: {repr(p1)}")
raw_clean = re.sub(r'[^a-z0-9]', '', p1.lower())
print(f"Raw Clean: {raw_clean[:100]}")
kw = 'acute-angled triangle'
kw_clean = re.sub(r'[^a-z0-9]', '', kw.lower())
print(f"KW Clean: {kw_clean}")
print(f"Match: {kw_clean in raw_clean}")
