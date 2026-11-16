import json

with open("DemoThermalModel.ipynb", "r", encoding="utf-8") as f:
    nb = json.load(f)

stubs = {
    2:  ["# Extracted to thermal/properties.py\n"],
    4:  ["# Extracted to thermal/correlations.py\n"],
    6:  ["# Extracted to thermal/correlations.py\n"],
    8:  ["# Extracted to thermal/components.py\n"],
    10: ["# Extracted to thermal/components.py\n"],
    12: ["# Extracted to thermal/simulation.py\n"],
    21: ["# Extracted to thermal/utils.py\n"],
    24: ["# Extracted to thermal/utils.py\n"],
}

# Add `from thermal import *` to the imports cell (cell 0)
imports_cell = nb["cells"][0]
src = imports_cell.get("source", [])
if not any("from thermal import" in line for line in src):
    # Add newline to last line if needed, then append the import
    if src and not src[-1].endswith("\n"):
        src[-1] += "\n"
    src.append("from thermal import *")
    imports_cell["source"] = src
    print("Cell 0: added 'from thermal import *'")

# Replace library cells with stubs
for idx, stub_src in stubs.items():
    nb["cells"][idx]["source"] = stub_src
    print(f"Cell {idx}: replaced with stub")

with open("DemoThermalModel.ipynb", "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print("Done.")
