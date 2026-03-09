import advanced_modules
import discovery_engine_v5 as eng
import re

advanced_modules.install()

problems = ["What is $1-1$?", "What is $0\\times10$?", "Solve $4+x=4$ for $x$."]
for p in problems:
    res = eng.run(p, quiet=False)
    # Check if we can extract the answer
    clean = re.sub(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])', '', str(res))
    m = re.search(r'FINAL ANSWER\n-+\n\s*(\d+)', clean)
    ans = m.group(1) if m else "NONE"
    print(f"\nPROB: {p} -> ANS: {ans}\n")
