p = 'simulacion/bioconversion_ode.py'
s = open(p).read()
if 'PREP = dict' in s:
    print('ya aplicado'); raise SystemExit

V1 = "def tasas_larva(B, t_d):"
N1 = ("PREP = dict(t_p=12.0, ancho=1.0, m_suelo=0.13)  # S5: switch prepupa (supuestos.md)\n\n"
      "def tasas_larva(B, t_d):")
assert s.count(V1) == 1, 'FALLO 1'
s = s.replace(V1, N1)

V2 = "a = PHY['amax']/(1.0+(B/Bmax)**PHY['alpha'])"
N2 = ("S_prep = 1.0/(1.0+np.exp(-(t_d-PREP['t_p'])/PREP['ancho']))  # S5\n"
      "    a = PHY['amax']/(1.0+(B/Bmax)**PHY['alpha'])*(1.0-S_prep)")
assert s.count(V2) == 1, 'FALLO 2'
s = s.replace(V2, N2)

V3 = "rCm = PHY['m']*B\n"
N3 = "rCm = PHY['m']*B*((1.0-S_prep)+PREP['m_suelo']*S_prep)\n"
assert s.count(V3) == 1, 'FALLO 3'
s = s.replace(V3, N3)

V4 = "DM_p = (-N*rA/1000.0 - kmic*DM)/DIA"
N4 = ("comida = 0.0 if DM <= 1e-9 else N*rA/1000.0        # S3: tope por DM\n"
      "    DM_p = (-comida - kmic*DM)/DIA")
assert s.count(V4) == 1, 'FALLO 4'
s = s.replace(V4, N4)

open(p, 'w').write(s)
print('patch OK')
