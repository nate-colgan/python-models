import json

with open("DemoThermalModel.ipynb", "r", encoding="utf-8") as f:
    nb = json.load(f)

for i, cell in enumerate(nb["cells"]):
    src = cell.get("source", [])
    first = src[0].strip()[:80] if src else "(empty)"
    print(f"Cell {i:3d}: {first}")
