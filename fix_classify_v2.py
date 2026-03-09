with open('discovery_engine_v5.py', 'r') as f:
    content = f.read()

# Remove the incorrectly placed AIMO check
aimo_block = """
    if any(kw in low for kw in ('triangle', 'perimeter', 'remainder', 'integer', 'function f', 'tournament', 'runners')):
        return Problem(raw=raw, ptype=PT.AIMO)"""
content = content.replace(aimo_block, "")

# Re-insert AIMO check after specific types but before generic ones
insertion_point = "if re.match(r'^(prove|show|demonstrate)\\b', low):"
aimo_fixed = """if any(kw in low for kw in ('triangle', 'perimeter', 'remainder', 'function f', 'tournament', 'runners')) or (('integer' in low or 'integer' in raw) and 'sum' not in low):
        return Problem(raw=raw, ptype=PT.AIMO)

    """
content = content.replace(insertion_point, aimo_fixed + insertion_point)

with open('discovery_engine_v5.py', 'w') as f:
    f.write(content)
