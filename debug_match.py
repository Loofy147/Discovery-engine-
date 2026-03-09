import re
import pandas as pd
df = pd.read_csv('reference.csv')
p1 = df.iloc[0]['problem']
print(f"P1: {p1[:50]}...")
pat = r'acute-angled triangle'
print(f"Search '{pat}': {bool(re.search(pat, p1))}")
print(f"Search '{pat}' lower: {bool(re.search(pat, p1.lower()))}")
