import csv
import subprocess
import time
import re
import advanced_modules
import discovery_engine_v5 as eng

# Install patches once
advanced_modules.install()

def run_engine(problem_text):
    start_time = time.time()
    try:
        res_raw = eng.run(problem_text, quiet=True)
        # Strip ANSI escape codes
        ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
        clean_stdout = ansi_escape.sub('', str(res_raw))

        # Extract FINAL ANSWER
        m = re.search(r'FINAL ANSWER\n-+\n\s*(\d+)', clean_stdout)
        final_answer = m.group(1).strip() if m else "No answer found"

        return {
            "status": "success",
            "answer": final_answer,
            "elapsed": time.time() - start_time
        }
    except Exception as e:
        return {
            "status": "error",
            "answer": str(e),
            "elapsed": time.time() - start_time
        }

def main():
    problems = []
    with open('reference.csv', mode='r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if i >= 5: break
            problems.append(row)

    print(f"{'ID':<10} | {'Expected':<10} | {'Result':<10} | {'Time':<10} | {'Status'}")
    print("-" * 60)

    for p in problems:
        res = run_engine(p['problem'])
        print(f"{p['id']:<10} | {p['answer']:<10} | {res['answer']:<10} | {res['elapsed']:.3f}s | {res['status']}")

if __name__ == "__main__":
    main()
