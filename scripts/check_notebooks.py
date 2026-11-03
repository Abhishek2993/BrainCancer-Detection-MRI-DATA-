"""CI check: every notebook is valid JSON, has no stored outputs, and its code cells parse.

Shell/magic lines (starting with ! or %) are skipped before parsing.
"""
import ast
import json
import sys
from pathlib import Path

failed = False
for nb_path in sorted(Path(sys.argv[1] if len(sys.argv) > 1 else ".").glob("*.ipynb")):
    nb = json.loads(nb_path.read_text())
    for i, cell in enumerate(nb["cells"]):
        if cell["cell_type"] != "code":
            continue
        src = "".join(cell["source"])
        if cell.get("outputs"):
            print(f"{nb_path}: cell {i} has stored outputs (clear them before committing)")
            failed = True
        code = "\n".join(l for l in src.splitlines() if not l.lstrip().startswith(("!", "%")))
        try:
            ast.parse(code)
        except SyntaxError as e:
            print(f"{nb_path}: cell {i} does not parse: {e}")
            failed = True
    print(f"{nb_path}: {len(nb['cells'])} cells checked")
sys.exit(1 if failed else 0)
