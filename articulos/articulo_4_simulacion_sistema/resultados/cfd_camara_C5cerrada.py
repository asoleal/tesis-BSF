#!/usr/bin/env python3
"""
cfd_camara.py - Verificacion CFD del supuesto de mezcla homogenea (A1).
BCs por subespacios (forma documentada dolfinx 0.11) + verificacion.
Configuracion de puertos por variables de entorno CFD_* (default C4).
Flujo: Stokes penalizado (LU). Transporte CO2: adv-dif + SUPG, Euler implicito.
"""
import os, sys
import numpy as np
import dolfinx
from dolfinx import mesh, fem, io
from mpi4py import MPI
import ufl
import gmsh

print("dolfinx", dolfinx.__version__)
try:
    from dolfinx.fem import LinearProblem
except ImportError:
    from dolfinx.fem.petsc import LinearProblem

IMG = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'imagenes')
os.makedirs(IMG, exist_ok=True)
CFG = os.environ.get("CFD_CONFIG", "C4")
RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'resultados')
os.makedirs(RES, exist_ok=True)
import shutil, datetime
shutil.copy(os.path.abspath(__file__), os.path.join(RES, f"cfd_camara_{CFG}.py"))

# ---------------- configuracion de puertos ----------------
Lx, Ly, Lz = 0.26, 0.19, 0.18
R_PUERTO = 0.003
V_AIRE = 2.5e-3
IN_Y = float(os.environ.get("CFD_IN_Y", "0.04"))
IN_Z = float(os.environ.get("CFD_IN_Z", "0.04"))
IN_TANG = os.environ.get("CFD_TANG", "1") == "1"     # componente tangencial +y
OUT_FACE = os.environ.get("CFD_OUT_FACE", "x")     # 'x' (pared x=Lx) o 'z' (tapa)
OUT_X = float(os.environ.get("CFD_OUT_X", "0.05")) # solo si OUT_FACE='z'
OUT_Y = float(os.environ.get("CFD_OUT_Y", "0.14")) # Ly-0.05
OUT_Z = float(os.environ.get("CFD_OUT_Z", "0.13")) # Lz-0.05
print(f"config {CFG}: IN=({IN_Y},{IN_Z}) tang={IN_TANG} | OUT face {OUT_FACE} ({OUT_X},{OUT_Y},{OUT_Z})")

# ---------------- malla gmsh: discos conformales ----------------
gmsh.initialize()
gmsh.model.add("camara")
gmsh.option.setNumber("Mesh.CharacteristicLengthMin", 1.0e-3)
gmsh.option.setNumber("Mesh.CharacteristicLengthMax", 12e-3)
box = gmsh.model.occ.addBox(0, 0, 0, Lx, Ly, Lz)
d_in = gmsh.model.occ.addDisk(-1.0e-3, IN_Y, IN_Z, R_PUERTO, R_PUERTO, -1, [1, 0, 0])
if OUT_FACE == "x":
    d_out = gmsh.model.occ.addDisk(Lx + 1.0e-3, OUT_Y, OUT_Z, R_PUERTO, R_PUERTO, -1, [1, 0, 0])
    p_out = gmsh.model.occ.addPoint(Lx, OUT_Y, OUT_Z, meshSize=1.0e-3)
else:
    d_out = gmsh.model.occ.addDisk(OUT_X, OUT_Y, Lz + 1.0e-3, R_PUERTO, R_PUERTO, -1, [0, 0, 1])
    p_out = gmsh.model.occ.addPoint(OUT_X, OUT_Y, Lz, meshSize=1.0e-3)
p_in = gmsh.model.occ.addPoint(0.0, IN_Y, IN_Z, meshSize=1.0e-3)
obj, mappa = gmsh.model.occ.fragment([(3, box)], [(2, d_in), (2, d_out), (0, p_in), (0, p_out)])
gmsh.model.occ.synchronize()
vols = [t[1] for t in gmsh.model.getEntities(3)]
gmsh.model.addPhysicalGroup(3, vols, 1)
gmsh.model.setPhysicalName(3, 1, "Camara")
in_ids = [t[1] for t in mappa[1]]
out_ids = [t[1] for t in mappa[2]]
print("puerto in areas:", [gmsh.model.occ.getMass(2, s) for s in in_ids])
print("puerto out areas:", [gmsh.model.occ.getMass(2, s) for s in out_ids])
gmsh.model.mesh.generate(3)
from dolfinx.io import gmsh as gmshio_mod
res = gmshio_mod.model_to_mesh(gmsh.model, MPI.COMM_WORLD, 0, gdim=3)
msh = res[0]
gmsh.finalize()
print(f"malla: {msh.topology.index_map(3).size_local} tets")

dx = ufl.dx(domain=msh)

# ---------------- marcado geometrico de facets ----------------
msh.topology.create_connectivity(2, 0)
conn20 = msh.topology.connectivity(2, 0)
coords = msh.geometry.x
todas = mesh.locate_entities_boundary(msh, 2, lambda x: np.full(x.shape[1], True))
facet_in, facet_out, facet_wall = [], [], []
for f in todas:
    mid = coords[conn20.links(f)].mean(axis=0)
    if abs(mid[0]) < 1e-6 and (mid[1]-IN_Y)**2 + (mid[2]-IN_Z)**2 < R_PUERTO**2:
        facet_in.append(f)
    elif OUT_FACE == "x" and abs(mid[0]-Lx) < 1e-6 and (mid[1]-OUT_Y)**2 + (mid[2]-OUT_Z)**2 < R_PUERTO**2:
        facet_out.append(f)
    elif OUT_FACE == "z" and abs(mid[2]-Lz) < 1e-6 and (mid[0]-OUT_X)**2 + (mid[1]-OUT_Y)**2 < R_PUERTO**2:
        facet_out.append(f)
    else:
        facet_wall.append(f)
facet_in = np.array(facet_in, dtype=np.int32)
facet_out = np.array(facet_out, dtype=np.int32)
facet_wall = np.array(facet_wall, dtype=np.int32)
print(f"facets: in={len(facet_in)} out={len(facet_out)} wall={len(facet_wall)}")

mt = mesh.meshtags(msh, 2,
                   np.concatenate([facet_in, facet_out, facet_wall]),
                   np.concatenate([np.full(len(facet_in), 2, np.int32),
                                   np.full(len(facet_out), 3, np.int32),
                                   np.full(len(facet_wall), 4, np.int32)]))

if os.environ.get("CFD_MALLA"):
    sys.exit(0)

# ---------------- marcadores geometricos para BCs ----------------
def es_frontera(x):
    return np.logical_or.reduce([
        np.isclose(x[0], 0.0, atol=1e-7), np.isclose(x[0], Lx, atol=1e-7),
        np.isclose(x[1], 0.0, atol=1e-7), np.isclose(x[1], Ly, atol=1e-7),
        np.isclose(x[2], 0.0, atol=1e-7), np.isclose(x[2], Lz, atol=1e-7)])
def inlet_m(x):
    return np.logical_and(np.isclose(x[0], 0.0, atol=1e-7),
           (x[1]-IN_Y)**2 + (x[2]-IN_Z)**2 < R_PUERTO**2)
def outlet_m(x):
    if OUT_FACE == "x":
        return np.logical_and(np.isclose(x[0], Lx, atol=1e-7),
               (x[1]-OUT_Y)**2 + (x[2]-OUT_Z)**2 < R_PUERTO**2)
    return np.logical_and(np.isclose(x[2], Lz, atol=1e-7),
           (x[0]-OUT_X)**2 + (x[1]-OUT_Y)**2 < R_PUERTO**2)
def wall_m(x):
    return np.logical_and(es_frontera(x),
           np.logical_not(np.logical_or(inlet_m(x), outlet_m(x))))

# ---------------- flujo (Stokes penalizado), BCs por subespacio ----------------
Q = 1.6667e-5
Q0 = os.environ.get("CFD_Q0") == "1"
U_IN = 0.0 if Q0 else Q/(np.pi*R_PUERTO**2)
FAN_ON = os.environ.get("CFD_FAN") == "1"
FAN_F = float(os.environ.get("CFD_FAN_F", "400"))
FAN_R = float(os.environ.get("CFD_FAN_R", "0.02"))
FAN_POS = (Lx/2.0, Ly/2.0, Lz - 0.03)
NPASOS_CFG = int(os.environ.get("CFD_NPASOS", "360"))
MU = 1.9e-5
LAMBDA = 1.0e4*MU

V = fem.functionspace(msh, ("Lagrange", 2, (3,)))
u, v_ = ufl.TrialFunction(V), ufl.TestFunction(V)
a_u = (MU*ufl.inner(ufl.grad(u), ufl.grad(v_))
       + LAMBDA*ufl.div(u)*ufl.div(v_))*dx
if FAN_ON:
    _x = ufl.SpatialCoordinate(msh)
    _r2 = (_x[0]-FAN_POS[0])**2 + (_x[1]-FAN_POS[1])**2 + (_x[2]-FAN_POS[2])**2
    _chi = ufl.conditional(ufl.lt(_r2, FAN_R**2), 1.0, 0.0)
    L_u = (-_chi*fem.Constant(msh, FAN_F)*v_[2]
           + ufl.inner(fem.Constant(msh, (0.0, 0.0, 0.0)), v_))*dx
    print(f"fan: F0={FAN_F} N/m3, esfera r={FAN_R} m en {FAN_POS}")
else:
    L_u = ufl.inner(fem.Constant(msh, (0.0, 0.0, 0.0)), v_)*dx
bcs_u = []
for _i in range(3):
    _d = fem.locate_dofs_geometrical((V.sub(_i), V), wall_m)
    bcs_u.append(fem.dirichletbc(fem.Constant(msh, 0.0), _d[0], V.sub(_i)))
_dix = fem.locate_dofs_geometrical((V.sub(0), V), inlet_m)
bcs_u.append(fem.dirichletbc(fem.Constant(msh, U_IN), _dix[0], V.sub(0)))
if IN_TANG:
    _diy = fem.locate_dofs_geometrical((V.sub(1), V), inlet_m)
    bcs_u.append(fem.dirichletbc(fem.Constant(msh, U_IN), _diy[0], V.sub(1)))
_ocomp = 0 if OUT_FACE == "x" else 2
_dox = fem.locate_dofs_geometrical((V.sub(_ocomp), V), outlet_m)
bcs_u.append(fem.dirichletbc(fem.Constant(msh, U_IN), _dox[0], V.sub(_ocomp)))
prob_u = LinearProblem(a_u, L_u, bcs=bcs_u,
                       petsc_options_prefix="stokes_",
                       petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
uh = prob_u.solve()
print(f"|u| max = {np.max(np.linalg.norm(uh.x.array.reshape(-1,3),axis=1)):.3f} m/s (U_IN={U_IN:.3f})")

# verificacion de restricciones
_v0 = uh.sub(0)
_chk_i = fem.locate_dofs_geometrical((V.sub(0), V), inlet_m)[0]
_chk_w = fem.locate_dofs_geometrical((V.sub(0), V), wall_m)[0]
print(f"verif BC: inlet ux=[{_v0.x.array[_chk_i].min():.3f},{_v0.x.array[_chk_i].max():.3f}] "
      f"pared ux=[{_v0.x.array[_chk_w].min():.1e},{_v0.x.array[_chk_w].max():.1e}]")

ds = ufl.Measure("ds", domain=msh, subdomain_data=mt)
n = ufl.FacetNormal(msh)
q_in = fem.assemble_scalar(fem.form(ufl.dot(uh, n)*ds(2)))
q_out = fem.assemble_scalar(fem.form(ufl.dot(uh, n)*ds(3)))
Q_eff = (abs(q_in)+q_out)/2.0
with open(os.path.join(RES, f"resumen_{CFG}.md"), "w") as _f:
    _f.write(f"# Configuracion {CFG} — {datetime.datetime.now():%Y-%m-%d %H:%M}\n\n")
    _f.write(f"- inlet (0, {IN_Y}, {IN_Z}) tang={IN_TANG}; outlet face {OUT_FACE} ({OUT_X}, {OUT_Y}, {OUT_Z})\n")
    _f.write(f"- caudal in = {abs(q_in)*60000:.3f} L/min, out = {q_out*60000:.3f} L/min, "
             f"Q_eff = {Q_eff*60000:.3f} L/min\n")
    _f.write(f"- |u| max = {np.max(np.linalg.norm(uh.x.array.reshape(-1,3),axis=1)):.3f} m/s "
             f"(U_IN = {U_IN:.3f} m/s)\n")
print(f"caudal in={abs(q_in)*60000:.3f} L/min | out={q_out*60000:.3f} L/min | Q_obj={Q*60000:.3f} L/min")

if os.environ.get("CFD_SOLO_FLUJO"):
    sys.exit(0)

vol = fem.assemble_scalar(fem.form(1.0*dx))
with io.VTXWriter(msh.comm, os.path.join(IMG, f"cfd_velocidad_{CFG}.bp"), [uh]) as f:
    f.write(0.0)

# ---------------- transporte de CO2 ----------------
D_CO2 = 2.0e-5
C = fem.functionspace(msh, ("Lagrange", 1))
c, w_ = ufl.TrialFunction(C), ufl.TestFunction(C)
c_prev = fem.Function(C)
c_prev.interpolate(lambda x: np.where(x[2] < 0.05, 1.0, 0.0))
bcs_c = []
if U_IN > 0:
    bcs_c.append(fem.dirichletbc(np.float64(0.0),
                fem.locate_dofs_geometrical(C, inlet_m), C))

DT, NPASOS = 2.0, NPASOS_CFG
h = ufl.CellDiameter(msh)
unorm = ufl.sqrt(ufl.dot(uh, uh))
tau = h/(2.0*unorm + h/DT)
res_trial = c/DT + ufl.dot(uh, ufl.grad(c)) - D_CO2*ufl.div(ufl.grad(c))
a_c = (c/DT*w_ + w_*ufl.dot(uh, ufl.grad(c))
       + D_CO2*ufl.dot(ufl.grad(c), ufl.grad(w_)))*dx \
      + tau*ufl.dot(uh, ufl.grad(w_))*res_trial*dx
L_c = (c_prev/DT*w_ + tau*ufl.dot(uh, ufl.grad(w_))*(c_prev/DT))*dx
prob_c = LinearProblem(a_c, L_c, bcs=bcs_c,
                       petsc_options_prefix="adv_",
                       petsc_options={"ksp_type": "preonly", "pc_type": "lu"})

masa0 = fem.assemble_scalar(fem.form(c_prev*dx))
media0 = masa0/vol
tiempo, eta_hist, masa_hist = [], [], []
tau_mix = None
for k in range(NPASOS):
    sol = prob_c.solve()
    c_prev.x.array[:] = sol.x.array
    masa = fem.assemble_scalar(fem.form(c_prev*dx))
    media = masa/vol
    var = fem.assemble_scalar(fem.form((c_prev-media)**2*dx))/vol
    eta = float(np.sqrt(var)/(media0 + 1e-12))
    t = (k+1)*DT
    tiempo.append(t); eta_hist.append(eta); masa_hist.append(masa)
    if tau_mix is None and eta < 0.05:
        tau_mix = t
    if k % 30 == 0:
        print(f"  t={t:6.1f} s  media={media:8.4f}  eta={eta:.4f}")

with io.VTXWriter(msh.comm, os.path.join(IMG, f"cfd_co2_final_{CFG}.bp"), [c_prev]) as f:
    f.write(0.0)

drift = 100.0*(masa_hist[-1]-masa0)/masa0
if Q_eff > 1e-12:
    tau_aire = V_AIRE/Q_eff
    prediccion = f"{100*(np.exp(-tiempo[-1]/tau_aire)-1):+.1f} %"
else:
    tau_aire = float('inf')
    prediccion = "n/a (camara cerrada)"
print(f"ideal bien mezclado: {prediccion}")
print(f"\nmasa: inicial={masa0:.4f} final={masa_hist[-1]:.4f} (deriva {drift:+.2f}%)")
with open(os.path.join(RES, f"resumen_{CFG}.md"), "a") as _f:
    _f.write(f"- tau_mix = {tau_mix if tau_mix else '>6 min'} s, tau_aire = {tau_aire:.0f} s, "
             f"relacion = {(tau_mix/tau_aire) if tau_mix else float('nan'):.2f}\n")
    _f.write(f"- masa: {drift:+.2f} % a {tiempo[-1]:.0f} s "
             f"(bien mezclado predice {100*(np.exp(-tiempo[-1]/tau_aire)-1):+.1f} %)\n")
with open(os.path.join(RES, f"hist_{CFG}.csv"), "w") as _f:
    _f.write("t_s,eta,masa\n")
    for _t, _e, _m in zip(tiempo, eta_hist, masa_hist):
        _f.write(f"{_t},{_e:.6f},{_m:.6e}\n")
print(f"tau_mix = {tau_mix if tau_mix else '>6 min'} s | tau_aire = {tau_aire:.0f} s | "
      f"relacion = {(tau_mix/tau_aire) if tau_mix else float('nan'):.2f}")

import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(7, 4))
ax.semilogy(np.array(tiempo), eta_hist, lw=1.5,
            label=r'$\eta(t)$ coef. variacion vs $c_0$')
ax2 = ax.twinx()
ax2.plot(np.array(tiempo), 100*(np.array(masa_hist)-masa0)/masa0,
         color='tab:red', lw=1, alpha=0.6, label='deriva de masa (%)')
ax2.set_ylabel('deriva masa (%)', color='tab:red')
ax.axhline(0.05, color='r', ls='--', label='umbral 5%')
if tau_mix:
    ax.axvline(tau_mix, color='g', ls=':', label=rf'$\tau_{{mix}}$ = {tau_mix:.0f} s')
if np.isfinite(tau_aire): ax.axvline(tau_aire, color='k', ls='--', alpha=0.5,
           label=rf'$\tau_{{aire}}$ = {tau_aire:.0f} s')
ax.set_xlabel('t (s)'); ax.set_ylabel(r'$\eta$')
ln, lb = ax.get_legend_handles_labels()
ln2, lb2 = ax2.get_legend_handles_labels()
ax.legend(ln+ln2, lb+lb2, fontsize=8, loc='lower left')
fig.tight_layout()
fig.savefig(os.path.join(IMG, f"cfd_tau_mix_{CFG}.pdf"))

def _idw_plano(coords, val, valor=Ly/2.0, tol=8e-3, nx=130, nz=90):
    if coords.shape[0] != val.size:
        coords = np.repeat(coords, val.size // coords.shape[0], axis=0)
    mask = np.abs(coords[:, 1] - valor) < tol
    P = coords[mask]; Vv = val[mask]
    xs = np.linspace(0, Lx, nx); zs = np.linspace(0, Lz, nz)
    X, Z = np.meshgrid(xs, zs)
    G = np.stack([X.ravel(), np.full(X.size, valor), Z.ravel()], axis=1)
    out = np.empty(G.shape[0])
    for i in range(0, G.shape[0], 400):
        d = np.linalg.norm(G[i:i+400, None, :] - P[None, :, :], axis=2)
        idx = np.argpartition(d, min(8, P.shape[0]-1), axis=1)[:, :8]
        dd = np.take_along_axis(d, idx, axis=1)
        w = 1.0/np.maximum(dd, 1e-9)**2
        out[i:i+400] = (w*Vv[idx]).sum(axis=1)/w.sum(axis=1)
    return X, Z, out.reshape(X.shape)

coordsV = V.tabulate_dof_coordinates()
modV = np.linalg.norm(uh.x.array.reshape(-1, 3), axis=1)
coordsC = C.tabulate_dof_coordinates()
XV, ZV, UV = _idw_plano(coordsV, modV)
XC, ZC, CC = _idw_plano(coordsC, c_prev.x.array.copy())
figd, axd = plt.subplots(1, 2, figsize=(11, 4))
im0 = axd[0].pcolormesh(XV, ZV, UV, shading='auto', cmap='viridis')
axd[0].set_title('|u| (m/s) — plano y = Ly/2'); figd.colorbar(im0, ax=axd[0])
im1 = axd[1].pcolormesh(XC, ZC, CC, shading='auto', cmap='inferno')
axd[1].set_title('c_CO2 final — plano y = Ly/2'); figd.colorbar(im1, ax=axd[1])
for _a in axd:
    _a.set_xlabel('x (m)'); _a.set_ylabel('z (m)')
    if FAN_ON:
        _a.plot(FAN_POS[0], FAN_POS[2], 'wo', ms=9, mfc='none', mew=2)
        _a.annotate('FAN', (FAN_POS[0]+0.006, FAN_POS[2]), color='w')
    _a.plot(0, IN_Z, 'c^', ms=8, mfc='none', mew=2); _a.annotate('IN', (0.004, IN_Z), color='c')
    if OUT_FACE == "x":
        _a.plot(Lx, OUT_Z, 'gs', ms=8, mfc='none', mew=2)
        _a.annotate('OUT', (Lx-0.034, OUT_Z), color='lime')
    else:
        _a.plot(OUT_X, Lz, 'gs', ms=8, mfc='none', mew=2)
        _a.annotate('OUT', (OUT_X, Lz-0.012), color='lime')
figd.tight_layout()
figd.savefig(os.path.join(IMG, f"cfd_distribucion_{CFG}.pdf"))
print(f"Figuras: cfd_tau_mix_{CFG}.pdf, cfd_distribucion_{CFG}.pdf | .bp: cfd_velocidad_{CFG}, cfd_co2_final_{CFG}")
