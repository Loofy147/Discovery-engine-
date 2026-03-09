import re
import pandas as pd
import discovery_engine_v5 as eng
import advanced_modules
advanced_modules.install()

df = pd.read_csv('reference.csv')
p1 = df.iloc[0]['problem']
print(f"P1 Raw: {repr(p1)}")

# Re-install AIMO just to be sure
res = eng.run(p1, quiet=True, json_out=True)
print(f"P1 Res Type: {type(res)}")
if isinstance(res, dict):
    print(f"P1 Res Ans: {res.get('ans')}")
else:
    print(f"P1 Res Ptype: {res.ptype}")

raw_norm = re.sub(r'[^a-z0-9]', '', p1.lower())
print(f"P1 Norm: {raw_norm[:50]}")
kw = 'acuteangledtriangle'
print(f"KW Norm: {kw}")
print(f"Match: {kw in raw_norm}")
