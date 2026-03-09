import pandas as pd
from aimo_final_robust_submission import AIMOSystem
df = pd.read_csv('reference.csv')
sys = AIMOSystem()
correct = 0
for i, row in df.iterrows():
    ans = sys.solve(row['problem'])
    if str(ans) == str(row['answer']): correct += 1
print(f"Reference Accuracy: {correct}/{len(df)}")
