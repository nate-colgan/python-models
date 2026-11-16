import json

with open('tinless system model.ipynb', encoding='utf-8') as f:
    nb = json.load(f)

new_src = """\
dT_colors    = plt.cm.plasma(np.linspace(0.15, 0.85, len(temp_range_vals)))
liner_styles = {False: '-', True: '--'}
liner_labels = {False: 'no liner', True: 'W liner'}

fig, axes = plt.subplots(4, 2, figsize=(12, 18))

ax_ar,  ax_rout    = axes[0]
ax_dT,  ax_pwr     = axes[1]
ax_eff, ax_min_pwr = axes[2]
ax_bl2, ax_bl3     = axes[3]

for ci, dT_range in enumerate(temp_range_vals):
    for liner_bool_ in liner_bool_vals:
        try:
            lbl    = f'ΔT={dT_range}K, {liner_labels[liner_bool_]}'
            kwargs = dict(color=dT_colors[ci], linestyle=liner_styles[liner_bool_],
                          marker='o', label=lbl)

            ax_ar.plot(      bandgap_vals, [sweep_results[(dT_range, bg, liner_bool_)]['area_ratio'] for bg in bandgap_vals], **kwargs)
            ax_rout.plot(    bandgap_vals, [sweep_results[(dT_range, bg, liner_bool_)]['r_out']      for bg in bandgap_vals], **kwargs)
            ax_dT.plot(      bandgap_vals, [sweep_results[(dT_range, bg, liner_bool_)]['peak_dT']    for bg in bandgap_vals], **kwargs)
            ax_pwr.plot(     bandgap_vals, [sweep_results[(dT_range, bg, liner_bool_)]['peak_power'] / (4 * _side_len * _L * (1 - _f_bare) * _f_active * 10) for bg in bandgap_vals], **kwargs)
            ax_min_pwr.plot( bandgap_vals, [sweep_results[(dT_range, bg, liner_bool_)]['min_power']  / (4 * _side_len * _L * (1 - _f_bare) * _f_active * 10) for bg in bandgap_vals], **kwargs)
            ax_eff.plot(     bandgap_vals, [sweep_results[(dT_range, bg, liner_bool_)]['efficiency'] for bg in bandgap_vals], **kwargs)
        except KeyError:
            pass

ax_ar.set_ylabel('Optimal area ratio')
ax_ar.set_title('Optimal area ratio')
ax_rout.set_ylabel('Outer radius [m]')
ax_rout.set_title('Block outer radius')
ax_dT.set_ylabel('Peak ΔT (inner − avg) [K]')
ax_dT.set_title('Peak inner wall dT')
ax_pwr.set_ylabel('Peak power density [W/cm²]')
ax_pwr.set_title('Peak power (start of discharge)')
ax_min_pwr.set_ylabel('Min power density [W/cm²]')
ax_min_pwr.set_title('Min power (end of discharge)')
ax_eff.set_ylabel('Overall efficiency [%]')
ax_eff.set_title('Cumulative elec / thermal')

for ax in [ax_ar, ax_rout, ax_dT, ax_pwr, ax_min_pwr, ax_eff]:
    ax.set_xlabel('Bandgap [μm]')
    ax.grid(True)

for ax in [ax_bl2, ax_bl3]:
    ax.set_visible(False)

ax_ar.legend(fontsize=9)
fig.suptitle(
    f'Parametric sweep  |  T_min={_T_min} K  dur={_duration_hr} hr  avg_flux={_avg_flux} W/cm²',
    fontsize=10,
)
fig.tight_layout()
plt.show()
"""

for cell in nb['cells']:
    if not cell.get('id', '').startswith('52d87076'):
        continue
    cell['source'] = new_src
    cell['outputs'] = []
    cell['execution_count'] = None
    print('Patched 52d87076')
    break

with open('tinless system model.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
print('Saved.')
