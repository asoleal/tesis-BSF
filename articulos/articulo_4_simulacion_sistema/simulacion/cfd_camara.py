#!/usr/bin/env python3
"""
cfd_camara.py - Verificacion CFD del supuesto de mezcla homogenea (A1).
Malla gmsh con refinamiento local en puertos (2 mm) y bulk (12 mm).
Flujo: Stokes penalizado. Transporte CO2: adveccion-difusion + SUPG,
Euler implicito con LU exacto. Se reporta tau_mix y balance de masa.
"""
import os
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

Lx, Ly, Lz = 0.26, 0.19, 0.18
R_PUERTO = 0.003
CENTRO_IN = (0.0, Ly/2.0, 0.05)
CENTRO_OUT = (Lx/2.0, Ly/2.0, Lz)

# ---------------- malla gmsh con refinamiento local ----------------
gmsh.initialize()
gmsh.model.add("camara")
gmsh.option.setNumber("Mesh.CharacteristicLengthMin", 2.0e-3)
gmsh.option.setNumber("Mesh.CharacteristicLengthMax", 12.0e-3)
gmsh.option.setNumber("Mesh.CharacteristicLengthFromPoints", 1)
box = gmsh.model.occ.addBox(0, 0, 0, Lx, Ly, Lz)
p_in = gmsh.model.occ.addPoint(*CENTRO_IN, meshSize=1.5e-3)
p_out = gmsh.model.occ.addPoint(*CENTRO_OUT, meshSize=1.5e-3)
gmsh.model.occ.fragment([(3, box)], [(0, p_in), (0, p_out)])
gmsh.model.occ.synchronize()
vols = gmsh.model.getEntities(3)
gmsh.model.addPhysicalGroup(3, [v[1] for v in vols], 1)
gmsh.model.setPhysicalName(3, 1, "Camara")
gmsh.model.mesh.generate(3)
from dolfinx.io import gmsh as gmshio_mod
res = gmshio_mod.model_to_mesh(gmsh.model, MPI.COMM_WORLD, 0, gdim=3)
msh = res[0]
gmsh.finalize()

# probe: distancia minima de centroides a cada puerto
msh.topology.create_connectivity(2, 0)
conn = msh.topology.connectivity(2, 0)
coords = msh.geometry.x
def _probe(nombre, facets, ij, ci, cj):
    dmin = 1e9
    for f in facets:
        mid = coords[conn.links(f)].mean(axis=0)
        d2 = (mid[ij[0]]-ci)**2 + (mid[ij[1]]-cj)**2
        dmin = min(dmin, d2)
    print(f"probe {nombre}: facets={len(facets)} dist2_min={dmin:.3e} (R^2={R_PUERTO**2:.3e})")
_probe("in", mesh.locate_entities_boundary(msh, 2, lambda x: np.isclose(x[0], 0.0, atol=1e-7)), (1, 2), Ly/2.0, 0.05)
_probe("out", mesh.locate_entities_boundary(msh, 2, lambda x: np.isclose(x[2], Lz, atol=1e-7)), (0, 1), Lx/2.0, Ly/2.0)

print(f"malla: {msh.topology.index_map(3).size_local} tets")

dx = ufl.dx(domain=msh)

def es_frontera(x):
    return np.logical_or.reduce([
        np.isclose(x[0], 0.0, atol=1e-7), np.isclose(x[0], Lx, atol=1e-7),
        np.isclose(x[1], 0.0, atol=1e-7), np.isclose(x[1], Ly, atol=1e-7),
        np.isclose(x[2], 0.0, atol=1e-7), np.isclose(x[2], Lz, atol=1e-7)])

def inlet_m(x):
    return np.logical_and(es_frontera(x),
           np.logical_and(np.isclose(x[0], 0.0, atol=1e-7),
           (x[1]-Ly/2.0)**2 + (x[2]-0.05)**2 < R_PUERTO**2))

def outlet_m(x):
    return np.logical_and(es_frontera(x),
           np.logical_and(np.isclose(x[2], Lz, atol=1e-7),
           (x[0]-Lx/2.0)**2 + (x[1]-Ly/2.0)**2 < R_PUERTO**2))

def wall_m(x):
    return np.logical_and(es_frontera(x),
           np.logical_not(np.logical_or(inlet_m(x), outlet_m(x))))

# ---------------- flujo (Stokes penalizado) ----------------
Q = 1.6667e-5
U_IN = Q/(np.pi*R_PUERTO**2)
MU = 1.9e-5
LAMBDA = 1.0e4*MU

V = fem.functionspace(msh, ("Lagrange", 2, (3,)))
u, v_ = ufl.TrialFunction(V), ufl.TestFunction(V)
a_u = (MU*ufl.inner(ufl.grad(u), ufl.grad(v_))
       + LAMBDA*ufl.div(u)*ufl.div(v_))*dx
L_u = ufl.inner(fem.Constant(msh, (0.0, 0.0, 0.0)), v_)*dx
for _n,_m in [("wall",wall_m),("inlet",inlet_m),("outlet",outlet_m)]:
    _d = fem.locate_dofs_geometrical(V,_m)
    _f = mesh.locate_entities_boundary(msh,2,_m)
    print(f"diag {_n}: dofs={len(_d)} facets={len(_f)}")
bc_w = fem.dirichletbc(fem.Constant(msh, (0.0, 0.0, 0.0)),
                       fem.locate_dofs_geometrical(V, wall_m), V)
bc_i = fem.dirichletbc(fem.Constant(msh, (U_IN, 0.0, 0.0)),
                       fem.locate_dofs_geometrical(V, inlet_m), V)
prob_u = LinearProblem(a_u, L_u, bcs=[bc_w, bc_i],
                       petsc_options_prefix="stokes_",
                       petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
uh = prob_u.solve()
print(f"|u| max = {np.max(np.linalg.norm(uh.x.array.reshape(-1,3),axis=1)):.3f} m/s (U_IN={U_IN:.3f})")

# verificacion de caudal en puertos
facetas_in = mesh.locate_entities_boundary(msh, 2, inlet_m)
facetas_out = mesh.locate_entities_boundary(msh, 2, outlet_m)
mt_in = mesh.meshtags(msh, 2, facetas_in, np.full(len(facetas_in), 1, np.int32))
mt_out = mesh.meshtags(msh, 2, facetas_out, np.full(len(facetas_out), 2, np.int32))
ds_in = ufl.Measure("ds", domain=msh, subdomain_data=mt_in)
ds_out = ufl.Measure("ds", domain=msh, subdomain_data=mt_out)
n = ufl.FacetNormal(msh)
q_in = fem.assemble_scalar(fem.form(ufl.dot(uh, n)*ds_in(1)))
q_out = fem.assemble_scalar(fem.form(ufl.dot(uh, n)*ds_out(2)))
import sys, os as _os
if _os.environ.get("CFD_SOLO_FLUJO"): sys.exit(0)
print(f"caudal in={q_in*60000:.2f} L/min | out={q_out*60000:.2f} L/min "
      f"| Q_obj={Q*60000:.2f} L/min")

vol = fem.assemble_scalar(fem.form(1.0*dx))
with io.VTXWriter(msh.comm, os.path.join(IMG, "cfd_velocidad.bp"), [uh]) as f:
    f.write(0.0)

# ---------------- transporte de CO2 ----------------
D_CO2 = 2.0e-5
C = fem.functionspace(msh, ("Lagrange", 1))
c, w_ = ufl.TrialFunction(C), ufl.TestFunction(C)
c_prev = fem.Function(C)
c_prev.interpolate(lambda x: np.where(x[2] < 0.05, 1.0, 0.0))
bc_c = fem.dirichletbc(np.float64(0.0),
                       fem.locate_dofs_geometrical(C, inlet_m), C)

DT, NPASOS = 2.0, 180
h = ufl.CellDiameter(msh)
unorm = ufl.sqrt(ufl.dot(uh, uh))
tau = h/(2.0*unorm + h/DT)
res_trial = c/DT + ufl.dot(uh, ufl.grad(c)) - D_CO2*ufl.div(ufl.grad(c))
a_c = (c/DT*w_ + w_*ufl.dot(uh, ufl.grad(c))
       + D_CO2*ufl.dot(ufl.grad(c), ufl.grad(w_)))*dx \
      + tau*ufl.dot(uh, ufl.grad(w_))*res_trial*dx
L_c = (c_prev/DT*w_ + tau*ufl.dot(uh, ufl.grad(w_))*(c_prev/DT))*dx
prob_c = LinearProblem(a_c, L_c, bcs=[bc_c],
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

with io.VTXWriter(msh.comm, os.path.join(IMG, "cfd_co2_final.bp"), [c_prev]) as f:
    f.write(0.0)

drift = 100.0*(masa_hist[-1]-masa0)/masa0
tau_aire = (2.5e-3)/Q
print(f"\nmasa: inicial={masa0:.4f} final={masa_hist[-1]:.4f} "
      f"(deriva {drift:+.2f}%)")
print(f"tau_mix = {tau_mix if tau_mix else '>6 min'} s | "
      f"tau_aire = {tau_aire:.0f} s | "
      f"relacion = {(tau_mix/tau_aire) if tau_mix else float('nan'):.2f}")

import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(7, 4))
ax.semilogy(np.array(tiempo), eta_hist, lw=1.5,
            label=r'$\eta(t)$ varianza normalizada')
ax2 = ax.twinx()
ax2.plot(np.array(tiempo), 100*(np.array(masa_hist)-masa0)/masa0,
         color='tab:red', lw=1, alpha=0.6, label='deriva de masa (%)')
ax2.set_ylabel('deriva masa (%)', color='tab:red')
ax.axhline(0.05, color='r', ls='--', label='umbral 5%')
if tau_mix:
    ax.axvline(tau_mix, color='g', ls=':', label=rf'$\tau_{{mix}}$ = {tau_mix:.0f} s')
ax.axvline(tau_aire, color='k', ls='--', alpha=0.5,
           label=rf'$\tau_{{aire}}$ = {tau_aire:.0f} s')
ax.set_xlabel('t (s)'); ax.set_ylabel(r'$\eta$')
ln, lb = ax.get_legend_handles_labels()
ln2, lb2 = ax2.get_legend_handles_labels()
ax.legend(ln+ln2, lb+lb2, fontsize=8, loc='lower left')
fig.tight_layout()
fig.savefig(os.path.join(IMG, "cfd_tau_mix.pdf"))
print("Figura: imagenes/cfd_tau_mix.pdf | Campos .bp listos para ParaView")
