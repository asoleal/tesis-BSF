p = 'simulacion/bioconversion_ode.py'
s = open(p).read()
if 'ppm_sm' in s:
    print('ya aplicado'); raise SystemExit
V = "        dk = min(max(margen/max(ppm_s, 1e-9), 120.0), 1800.0)\n"
N = ("        dk = min(max(margen/max(ppm_s, 1e-9), 120.0), 1800.0)\n"
     "        tm = t + dk/2.0\n"
     "        Ym = sol.sol(tm)\n"
     "        rm = tasas_larva(Ym[1], tm/DIA)\n"
     "        rmm = tasas_mic(Ym[4], Ym[5]/max(Ym[4] + Ym[5], 1e-9), Ym[6], Ym[3], rm[0])\n"
     "        ppm_sm = (Ym[3]*rm[3] + rmm[0]*1000.0)/44000.0/FIS['Vair']/DIA*FIS['R']*(Ym[7] + 273.15)/FIS['P']*1e6\n"
     "        dk = min(max(margen/max(ppm_sm, 1e-9), 120.0), 1800.0)\n")
assert s.count(V) == 1, 'FALLO: ancla no unica'
s = s.replace(V, N)
open(p, 'w').write(s)
print('patch OK')
