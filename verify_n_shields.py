import json
with open('tinless system model.ipynb', encoding='utf-8') as f:
    nb = json.load(f)

for cell in nb['cells']:
    cid = cell.get('id','')
    if not (cid.startswith('b2e4f810') or cid.startswith('c7d9e412')):
        continue
    src = cell['source'] if isinstance(cell['source'], str) else ''.join(cell['source'])
    lines = src.splitlines()
    print(f"\n=== {cid[:12]} ===")
    for i, l in enumerate(lines, 1):
        if any(kw in l for kw in ['nominal_cell_flux', 'N_shields', 'avg_flux']):
            print(f"  L{i:3}: {l.rstrip()}")
