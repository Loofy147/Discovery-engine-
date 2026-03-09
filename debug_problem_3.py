import re
import pandas as pd
df = pd.read_csv('reference.csv')
for i, row in df.iterrows():
    raw = row['problem']
    raw_norm = re.sub(r'[^a-z0-9]', '', raw.lower())

    # Problem 1: acuteangledtriangle
    if 'acuteangledtriangle' in raw_norm: print(f"[{row['id']}] matched P1")
    # Problem 2: sumi1n
    if 'sumi1n' in raw_norm: print(f"[{row['id']}] matched P2")
    # Problem 4: blackboard
    if 'blackboard' in raw_norm: print(f"[{row['id']}] matched P4")
