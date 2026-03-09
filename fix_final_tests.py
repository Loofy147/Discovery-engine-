import re

with open('discovery_engine_v5.py', 'r') as f:
    content = f.read()

# Fix the duplicate/broken classification
content = content.replace("if any(kw in low for kw in ('triangle', 'perimeter', 'remainder', 'integer', 'function f', 'tournament', 'runners', 'what is', 'solve', 'calculate')):\n        return Problem(raw=raw, ptype=PT.AIMO)", "")

# Move PT.SUM to higher priority
sum_block = """    if any(kw in low for kw in ("sum of", "1+2+", "series", "summation", "sigma")):
        return Problem(raw=raw, ptype=PT.SUM, meta={"summand": _parse_summand(raw)})"""

content = content.replace(sum_block, "")
content = "    " + sum_block + "\n" + content.replace("def classify(raw: str) -> Problem:\n    low = raw.lower().strip()", "def classify(raw: str) -> Problem:\n    low = raw.lower().strip()")

# Actually just insert it right after low = ...
content = content.replace("low = raw.lower().strip()", "low = raw.lower().strip()\n" + sum_block)

with open('discovery_engine_v5.py', 'w') as f:
    f.write(content)
