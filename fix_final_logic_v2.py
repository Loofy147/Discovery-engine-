import re

with open('discovery_engine_v5.py', 'r') as f:
    content = f.read()

# Make AIMO classification even broader for natural language math
aimo_block = """
    # AIMO specific patterns (Olympiad level)
    aimo_kws = ('triangle', 'perimeter', 'remainder', 'function f', 'tournament', 'runners', 'blackboard', 'n-tastic', 'sweets', 'norwegian', 'rectangles', 'shifty')
    if any(kw in low for kw in aimo_kws) or (('integer' in low or 'integer' in raw) and not any(skip in low for skip in ('sum', 'series', 'matrix', 'graph'))):
        return Problem(raw=raw, ptype=PT.AIMO)
"""

# Ensure it doesn't duplicate and is in the right place
content = re.sub(r'# AIMO specific patterns.*?return Problem\(raw=raw, ptype=PT\.AIMO\)', '', content, flags=re.DOTALL)
insertion_point = "e = _parse(raw)"
content = content.replace(insertion_point, aimo_block + "\n    " + insertion_point)

with open('discovery_engine_v5.py', 'w') as f:
    f.write(content)
