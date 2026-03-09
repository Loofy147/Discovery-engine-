import advanced_modules
import discovery_engine_v5 as eng

# Install patches
advanced_modules.install(verbose=True)

# Test a triangle problem
problem = "Let $ABC$ be an acute-angled triangle with side lengths and perimeter. Find the remainder when abc is divided by 10^5."
print(f"Solving: {problem}\n")
res = eng.run(problem)
