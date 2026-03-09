import os
import sys
import time
import json
import pandas as pd
import numpy as np
import sympy as sp
from datetime import datetime
import re

# Import our components
import discovery_engine_v5 as eng
import advanced_modules
import integrated_synthesis_engine as synthesis

def hr(ch="=", n=70): return ch * n

class AIMOSystem:
    def __init__(self):
        print(f"\n{hr()}")
        print("  AIMO FULL SYSTEM INTEGRATION — Discovery + Synthesis + DEGF")
        print(f"  Timestamp: {datetime.now().isoformat()}")
        print(hr())

        # Phase 1: Install advanced modules
        print("\n[Phase 1] Initializing Engine & Advanced Modules...")
        advanced_modules.install(verbose=True)
        self.engine = eng

    def benchmark(self, problems_csv='reference.csv', n=5):
        print(f"\n[Phase 2] Benchmarking on {n} reference problems...")
        df = pd.read_csv(problems_csv)
        results = []

        for i, row in df.iterrows():
            if i >= n: break
            prob_id = row['id']
            expected = str(row['answer'])

            print(f"  [{prob_id}] Solving...")
            start = time.time()
            # Run engine with DEGF tracking
            res_obj = self.engine.run(row['problem'], quiet=True, json_out=True)
            elapsed = time.time() - start

            # Extract answer
            if isinstance(res_obj, dict):
                if 'ans' in res_obj:
                    got = str(res_obj['ans'])
                elif 'solutions' in res_obj and res_obj['solutions']:
                    got = str(res_obj['solutions'][0])
                else:
                    got = '0'
                g_score = res_obj.get('degf_G', 0.5276) # default for AIMO heuristics
            else:
                got = '0'
                g_score = 0.0

            correct = (got == expected)

            results.append({
                "id": prob_id,
                "expected": expected,
                "got": got,
                "correct": correct,
                "time": elapsed,
                "degf_G": g_score
            })

            status = "✓" if correct else "✗"
            print(f"    Result: {got} (Exp: {expected}) {status} in {elapsed:.3f}s [G={g_score:.4f}]")

        avg_time = sum(r['time'] for r in results) / len(results)
        accuracy = sum(1 for r in results if r['correct']) / len(results)
        print(f"\n  Benchmark Complete: Accuracy {accuracy:.0%}, Avg Time {avg_time:.3f}s")
        return results

    def run_inference(self, test_csv='test.csv'):
        print(f"\n[Phase 3] Running inference on {test_csv}...")
        df_test = pd.read_csv(test_csv)
        ids, answers = [], []

        for _, row in df_test.iterrows():
            res = self.engine.run(row['problem'], quiet=True, json_out=True)
            if isinstance(res, dict):
                if 'ans' in res:
                    ans_str = res['ans']
                elif 'solutions' in res and res['solutions']:
                    ans_str = res['solutions'][0]
                else:
                    ans_str = '0'
            else:
                ans_str = '0'

            # Extract first integer from ans_str
            m = re.search(r'(-?\d+)', str(ans_str))
            ans = int(m.group(1)) if m else 0

            ids.append(row['id'])
            answers.append(ans)

        df_sub = pd.DataFrame({'id': ids, 'answer': answers})
        df_sub.to_parquet('submission.parquet', engine='pyarrow')
        print(f"  Inference Complete. Generated submission.parquet ({len(df_sub)} rows)")
        return df_sub

    def self_assessment(self, bench_results):
        print("\n[Phase 4] Meta Self-Assessment (DEGF Analytics)...")
        g_scores = [r['degf_G'] for r in bench_results]
        V = float(np.var(g_scores)) if g_scores else 0.0
        # Count "collapses" in performance or confidence
        C = sum(1 for i in range(1, len(g_scores)) if g_scores[i] < g_scores[i-1] - 0.05)

        meta_G = synthesis.G_degf(V, C/max(len(g_scores),1))
        print(f"  Meta-G: {meta_G:.4f}")
        print(f"  System Status: {'GENUINE' if meta_G > 0.7 else 'PATTERN_MATCHING'}")

        return {"meta_G": meta_G, "V": V, "C": C}

if __name__ == "__main__":
    system = AIMOSystem()
    bench = system.benchmark(n=10) # Test more
    system.self_assessment(bench)
    system.run_inference()
    print(f"\n{hr()}")
    print("  AIMO FULL SYSTEM RUN COMPLETE")
    print(f"{hr()}\n")
