import discovery_engine_v5 as eng
import advanced_modules
import integrated_synthesis_engine as synthesis
advanced_modules.install()
synthesis.attach_degf_to_discovery_engine(eng)
res = eng.run('Let $ABC$ be an acute-angled triangle', quiet=True, json_out=True)
print(f"Ans: {res.get('ans')}, G: {res.get('degf_G', 0):.4f}")
