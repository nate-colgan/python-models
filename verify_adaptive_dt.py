import json, sys
with open('tinless system model.ipynb', encoding='utf-8') as f:
    nb = json.load(f)

print("=== YAML dt fields ===")
for cell in nb['cells']:
    if not cell.get('id','').startswith('a8c3d501'):
        continue
    src = ''.join(cell['source']) if isinstance(cell['source'], list) else cell['source']
    for l in src.splitlines():
        if any(kw in l for kw in ['dt_short','dt_medium','dt_long','dt_short_dur',"'dt'"]):
            print(' ', l)
    break

print("\n=== run_simulation signature + timestep logic ===")
for cell in nb['cells']:
    if not cell.get('id','').startswith('03dd9a23'):
        continue
    src = ''.join(cell['source']) if isinstance(cell['source'], list) else cell['source']
    lines = src.splitlines()
    start = next(i for i,l in enumerate(lines) if 'def run_simulation(' in l)
    # print first 40 lines of method
    for i, l in enumerate(lines[start:start+50], start+1):
        sys.stdout.buffer.write((f'L{i:3}: {l}\n').encode('utf-8'))
    break

print("\n=== b2e4f810 dt reads + run call ===")
for cell in nb['cells']:
    if not cell.get('id','').startswith('b2e4f810'):
        continue
    src = ''.join(cell['source']) if isinstance(cell['source'], list) else cell['source']
    for i,l in enumerate(src.splitlines(),1):
        if any(kw in l for kw in ['dt_short','dt_medium','dt_long','run_simulation','snap']):
            sys.stdout.buffer.write((f'L{i:3}: {l}\n').encode('utf-8'))
    break

print("\n=== plot cell log scale ===")
for cell in nb['cells']:
    if not cell.get('id','').startswith('35eeb2ae'):
        continue
    src = ''.join(cell['source']) if isinstance(cell['source'], list) else cell['source']
    for i,l in enumerate(src.splitlines(),1):
        if 'xscale' in l or 'xlabel' in l:
            print(f'  L{i:3}: {l.rstrip()}')
    break
