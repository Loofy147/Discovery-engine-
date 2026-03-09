import csv
import re
import advanced_modules
import discovery_engine_v5 as eng

# Install patches
advanced_modules.install(verbose=False)

def solve(problem_text):
    try:
        # Run engine (will use AIMO solver internally)
        res_raw = eng.run(problem_text, quiet=True)
        # Strip ANSI
        ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
        clean_raw = ansi_escape.sub('', str(res_raw))

        # Extract from the AIMO-specific output format
        # It looks like:
        # FINAL ANSWER
        # ------------------------------------------------------------------------
        #   336
        m = re.search(r'FINAL ANSWER\n-+\n\s*(\d+)', clean_raw)
        if m:
            return int(m.group(1))

        # Try the original engine's final answer if AIMO solver didn't catch it
        # (Though run_advanced returns its own string for PT.AIMO)
        return 0
    except Exception as e:
        return 0

with open('test.csv', 'r') as fin:
    reader = csv.DictReader(fin)
    with open('submission.csv', 'w') as fout:
        writer = csv.DictWriter(fout, fieldnames=['id', 'answer'])
        writer.writeheader()
        for row in reader:
            ans = solve(row['problem'])
            writer.writerow({'id': row['id'], 'answer': ans})

print("Submission generated: submission.csv")
