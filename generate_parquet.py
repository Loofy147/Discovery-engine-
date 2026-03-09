import pandas as pd
import csv
import re
import advanced_modules
import discovery_engine_v5 as eng

# Install patches
advanced_modules.install(verbose=False)

def solve(problem_text):
    try:
        # Run engine
        res_raw = eng.run(problem_text, quiet=True)
        # Strip ANSI
        ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
        clean_raw = ansi_escape.sub('', str(res_raw))

        # Extract answer
        m = re.search(r'FINAL ANSWER\n-+\n\s*(\d+)', clean_raw)
        if m:
            return int(m.group(1))
        return 0
    except:
        return 0

# Read test.csv
df_test = pd.read_csv('test.csv')
ids = []
answers = []

for _, row in df_test.iterrows():
    ans = solve(row['problem'])
    ids.append(row['id'])
    answers.append(ans)

# Create submission dataframe
df_sub = pd.DataFrame({'id': ids, 'answer': answers})

# Save as parquet
df_sub.to_parquet('submission.parquet', engine='pyarrow')
print("Generated submission.parquet")
print(df_sub)
