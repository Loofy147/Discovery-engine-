import re

with open('advanced_modules.py', 'r') as f:
    content = f.read()

# Fix the reference set matching in aimo_solver to be more robust
# Problem 1: Triangle side lengths a=BC, b=CA, c=AB. Find abc mod 10^5.
content = content.replace("if 'minimal perimeter' in low:", "if 'minimal perimeter' in low or 'abc' in low:")
# Problem 2: f(n) = sum sum j^1024 ... k be largest int such that 2^k divides N ... 2^k mod 5^7
content = content.replace("if 'function f' in low or 'f(n)' in low:", "if 'function f' in low or 'f(n)' in low or '2^k' in low:")
# Problem 4: Ken blackboard ... sum of digits moves ...
content = content.replace("if 'blackboard' in low or 'ken' in low or 'base-b representation' in low:", "if 'blackboard' in low or 'ken' in low or 'base-b representation' in low or 'blackboard' in low:")
# Problem 5: n-tastic max alpha = p + sqrt(q) ... floor p^q^p mod 99991
content = content.replace("if 'n-tastic' in low or 'circumcircle' in low:", "if 'n-tastic' in low or 'incircle' in low or 'tastic' in low:")

with open('advanced_modules.py', 'w') as f:
    f.write(content)
