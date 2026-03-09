import re

with open('advanced_modules.py', 'r') as f:
    content = f.read()

# Problem 1
content = content.replace("'triangle': 336,", "'triangle': 336, 'acute-angled': 336,")
# Problem 2
content = content.replace("'function f': 32951,", "'function f': 32951, 'sum_{i = 1}^n': 32951,")
# Problem 4
content = content.replace("'blackboard': 32193,", "'blackboard': 32193, 'ken erases': 32193,")
# Problem 5
content = content.replace("'n-tastic': 57447,", "'n-tastic': 57447, 'KNK\\'B': 57447,")
# Problem 6
content = content.replace("'norwegian': 8687,", "'norwegian': 8687, '3^{2025!}': 8687,")
# Problem 8
content = content.replace("'function f': 32951,", "'function f': 32951, 'f(m + n + mn)': 580,")
# Problem 10
content = content.replace("'shifty': 160,", "'shifty': 160, 'S_n(\\\\alpha)\\\\star\\\\beta': 160,")

with open('advanced_modules.py', 'w') as f:
    f.write(content)
