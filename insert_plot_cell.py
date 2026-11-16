import json

with open('tinless system model.ipynb', encoding='utf-8') as f:
    nb = json.load(f)

src = [
    "from scipy.integrate import cumulative_trapezoid\n",
    "\n",
    "time_hr = results['time'] / 3600\n",
    "\n",
    "# -- Power vs time ------------------------------------------------------------\n",
    "fig1, ax1 = plt.subplots(figsize=(10, 4))\n",
    "ax1.plot(time_hr, results['Q_rad'] / 1e3, label='Thermal (Q_rad)', color='C1')\n",
    "ax1.plot(time_hr, results['P_elec'] / 1e3, label='Electrical', color='C0')\n",
    "ax1.set_xlabel('Time [hr]')\n",
    "ax1.set_ylabel('Power [kW]')\n",
    "ax1.set_title('Power vs Time')\n",
    "ax1.legend()\n",
    "ax1.grid(True)\n",
    "fig1.tight_layout()\n",
    "\n",
    "# -- Block temperature vs time ------------------------------------------------\n",
    "fig2, ax2 = plt.subplots(figsize=(10, 4))\n",
    "ax2.plot(time_hr, results['T_inner'] - 273.15, label='Inner wall', color='C3')\n",
    "ax2.plot(time_hr, results['T_avg']   - 273.15, label='Average',    color='C2')\n",
    "ax2.plot(time_hr, results['T_outer'] - 273.15, label='Outer wall', color='C0')\n",
    "ax2.set_xlabel('Time [hr]')\n",
    "ax2.set_ylabel('Temperature [C]')\n",
    "ax2.set_title('Block Temperature vs Time')\n",
    "ax2.legend()\n",
    "ax2.grid(True)\n",
    "fig2.tight_layout()\n",
    "\n",
    "# -- Cumulative electrical / thermal energy -----------------------------------\n",
    "cum_Q = cumulative_trapezoid(results['Q_rad'],  results['time'], initial=0)\n",
    "cum_P = cumulative_trapezoid(results['P_elec'], results['time'], initial=0)\n",
    "with np.errstate(divide='ignore', invalid='ignore'):\n",
    "    cum_eff = np.where(cum_Q > 0, cum_P / cum_Q * 100, 0.0)\n",
    "\n",
    "fig3, ax3 = plt.subplots(figsize=(10, 4))\n",
    "ax3.plot(time_hr, cum_eff, color='C4')\n",
    "ax3.set_xlabel('Time [hr]')\n",
    "ax3.set_ylabel('Cumulative efficiency [%]')\n",
    "ax3.set_title('Cumulative electrical / thermal energy')\n",
    "ax3.grid(True)\n",
    "fig3.tight_layout()\n",
    "\n",
    "total_Q = cum_Q[-1]\n",
    "total_P = cum_P[-1]\n",
    'print(f"Total thermal energy absorbed:   {total_Q/3.6e9:.3f} MWhr")\n',
    'print(f"Total electrical energy generated: {total_P/3.6e9:.3f} MWhr")\n',
    'print(f"Overall efficiency:              {total_P/total_Q*100:.2f}%")\n',
    "\n",
    "plt.show()\n",
]

# Replace the trailing empty cell
for cell in nb['cells']:
    if cell.get('id', '').startswith('35eeb2ae'):
        cell['source'] = src
        cell['outputs'] = []
        cell['execution_count'] = None
        print(f'Replaced empty cell 35eeb2ae with plot cell')
        break

with open('tinless system model.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
print('Done.')
