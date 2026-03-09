import os, pandas as pd, re
import discovery_engine_v5 as eng
import advanced_modules
import integrated_synthesis_engine as synthesis

advanced_modules.install()
synthesis.attach_degf_to_discovery_engine(eng)

df = pd.read_csv('reference.csv')
correct = 0
for i, row in df.iterrows():
    # Use the same logic as submission.py
    result = eng.run(row['problem'], quiet=True, json_out=True)

    ans = '0'
    if isinstance(result, dict):
        ans = str(result.get('ans', '0'))

    exp = str(row['answer'])
    if ans == exp: correct += 1
    print(f"[{row['id']}] Got: {ans}, Exp: {exp} {'✓' if ans == exp else '✗'}")

print(f"\nTotal Accuracy: {correct/len(df):.0%}")
