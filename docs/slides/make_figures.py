"""Publication-style figures for the qslab slides (all computed with qslab)."""
import os, itertools, time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import qslab
from qslab import lattice, models, ED
from qslab.kitaev import KitaevMajorana, kitaev_bloch, kitaev_phase, majorana_monte_carlo
from qslab.stabilizer import toric_code_ground_state, kitaev_preskill_regions
from qslab.parton import SpinonAnsatz, GutzwillerVMC
from qslab.nqs import NQS
from qslab.dmrg import DMRG
from qslab.ed import ftlm_thermodynamics

OUT = "figs"
NAVY, TEAL, ORANGE, RED, PURPLE, GREEN, GREY = "#23373B", "#14857A", "#EB811B", "#C0392B", "#7D3C98", "#2E8B57", "#8C8C8C"
plt.rcParams.update({"font.size": 11, "axes.labelsize": 12, "axes.titlesize": 12, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.linewidth": 0.9, "lines.linewidth": 2.0,
                     "legend.frameon": False, "figure.dpi": 200, "savefig.bbox": "tight",
                     "mathtext.fontset": "cm", "font.family": "DejaVu Sans"})
CM = LinearSegmentedColormap.from_list("q", ["#FAFAFA", "#9FD3CC", TEAL, NAVY])

def save(fig, name):
    fig.savefig(os.path.join(OUT, name + ".pdf")); fig.savefig(os.path.join(OUT, name + ".png")); plt.close(fig)
    print("saved", name, flush=True)

# 1. lattice gallery -----------------------------------------------------------
def draw(ax, lat, title, kc=None):
    for b in lat.bonds(1):
        p = lat.positions[b.i]; q = p + np.array(b.d)
        c = (kc or {}).get(b.kind, NAVY)
        ax.plot([p[0], q[0]], [p[1], q[1]], color=c, lw=2.2 if kc else 1.6, alpha=0.35 if b.wraps else 1,
                ls="--" if b.wraps else "-", solid_capstyle="round")
    ax.scatter(*lat.positions.T, s=28, c=[[0.9, 0.9, 0.9]], edgecolors=NAVY, zorder=3, lw=1.1)
    ax.set_aspect("equal"); ax.axis("off"); ax.set_title(title)
fig, ax = plt.subplots(1, 4, figsize=(13, 3.4))
draw(ax[0], lattice.honeycomb(4, 3), "honeycomb (Kitaev x, y, z)", {"x": RED, "y": GREEN, "z": TEAL})
draw(ax[1], lattice.kagome(3, 3), "kagome")
draw(ax[2], lattice.triangular(4, 4), "triangular")
draw(ax[3], lattice.honeycomb(6, 2, bc="cylinder"), "honeycomb cylinder (DMRG)", {"x": RED, "y": GREEN, "z": TEAL})
save(fig, "lattices")

# 2. Kitaev: bands + phase diagram ----------------------------------------------
lat1 = lattice.honeycomb(1, 1)
ks, dist, ticks = lat1.kpath(["G", "K", "M", "G"], n=150)
fig, ax = plt.subplots(1, 2, figsize=(11, 4.2), gridspec_kw={"width_ratios": [1.25, 1]})
for (K, kap, lab, c) in [((1, 1, 1), 0, r"isotropic, $\kappa=0$ (gapless)", NAVY),
                         ((1, 1, 1), 0.1, r"$\kappa=0.1$ (chiral, $\nu=\pm1$)", ORANGE),
                         ((0.3, 0.3, 1), 0, r"$K_z>K_x+K_y$ (A phase)", TEAL)]:
    e = np.array([np.linalg.eigvalsh(kitaev_bloch(k, K, kap))[1] for k in ks])
    ax[0].plot(dist, e, color=c, label=lab); ax[0].plot(dist, -e, color=c)
ax[0].set_xticks(ticks, [r"$\Gamma$", "K", "M", r"$\Gamma$"]); ax[0].axhline(0, c=GREY, lw=0.6)
for t in ticks: ax[0].axvline(t, c=GREY, lw=0.4)
ax[0].set_ylabel(r"Majorana energy $\varepsilon(\mathbf{k})$"); ax[0].legend(fontsize=9, loc="upper right")
n = 140; img = np.full((n, n), np.nan); B = lat1.reciprocal()
grid = [(a / 20) * B[0] + (b / 20) * B[1] for a in range(20) for b in range(20)]
for a in range(n):
    for b in range(n - a):
        Kx, Ky = a / n, b / n; Kz = 1 - Kx - Ky
        img[a, b] = 0 if kitaev_phase((Kx, Ky, Kz)).startswith("B") else min(np.linalg.eigvalsh(kitaev_bloch(k, (Kx, Ky, Kz)))[1] for k in grid)
im = ax[1].imshow(img, origin="lower", extent=[0, 1, 0, 1], cmap=CM)
ax[1].plot([0, .5, .5, 0], [.5, 0, .5, .5], c=ORANGE, lw=1.4)
ax[1].text(0.25, 0.3, "B", fontsize=20, color=ORANGE, weight="bold"); ax[1].text(0.06, 0.82, r"A$_x$", fontsize=13, color=NAVY)
ax[1].text(0.78, 0.07, r"A$_y$", fontsize=13, color=NAVY); ax[1].text(0.03, 0.05, r"A$_z$", fontsize=13, color=NAVY)
ax[1].set_xlabel(r"$K_y$"); ax[1].set_ylabel(r"$K_x$"); plt.colorbar(im, ax=ax[1], label="Majorana gap")
save(fig, "kitaev_bands_phase")

# 3. Kitaev: Majorana vs ED spectrum (8 sites) ---------------------------------
lat8 = lattice.honeycomb(2, 2)
fig, ax = plt.subplots(figsize=(5.6, 4))
for x0, kap, col in [(0, 0.0, NAVY), (1.6, 0.2, ORANGE)]:
    km = KitaevMajorana(lat8, (1, 0.7, 1.2) if kap else (1, 1, 1), kap); lv = set()
    for bits in itertools.product([1, -1], repeat=km.nb):
        u = np.array(bits, float); eps = km.spectrum(u); P = km.is_physical(u); E0 = -0.5 * eps.sum()
        for occ in itertools.product([0, 1], repeat=len(eps)):
            if (sum(occ) % 2 == 0) == P: lv.add(round(E0 + np.dot(occ, eps), 7))
    w = np.linalg.eigvalsh(models.kitaev(lat8, (1, 0.7, 1.2) if kap else (1, 1, 1), kappa=kap).to_matrix(8))
    for e in np.unique(np.round(w, 7)): ax.plot([x0, x0 + 0.6], [e, e], c=col, lw=1.4)
    for e in lv: ax.plot([x0 + 0.75, x0 + 1.35], [e, e], c=col, lw=1.4, alpha=0.55)
ax.set_xticks([0.3, 1.05, 1.9, 2.65], ["ED", "Majorana", "ED", "Majorana"])
ax.text(0.68, 7.6, r"$\kappa=0$", ha="center", color=NAVY); ax.text(2.28, 7.6, r"$\kappa=0.2$, anisotropic", ha="center", color=ORANGE)
ax.set_ylabel("energy levels (8 sites, 256 states)"); ax.spines["bottom"].set_visible(False)
save(fig, "kitaev_validation")

# 4. Kitaev thermodynamics (Majorana MC) ----------------------------------------
Ts = np.geomspace(0.01, 10, 28)
mc = majorana_monte_carlo(L=4, temperatures=Ts, n_therm=120, n_meas=260, seed=4)
fig, ax = plt.subplots(1, 3, figsize=(13, 3.5))
ax[0].semilogx(mc["T"], mc["C"], "o-", c=NAVY, ms=4); ax[0].set_ylabel(r"$C/N$")
ax[1].semilogx(mc["T"], mc["S"] / np.log(2), "o-", c=TEAL, ms=4); ax[1].axhline(0.5, ls="--", c=ORANGE); ax[1].set_ylabel(r"$S/(N\ln 2)$")
ax[1].text(0.35, 0.42, r"$\frac{1}{2}\ln 2$ plateau", color=ORANGE)
ax[2].semilogx(mc["T"], mc["W"], "o-", c=PURPLE, ms=4); ax[2].set_ylabel(r"$\langle W_p\rangle$")
for a in ax: a.set_xlabel(r"$T/K$")
for a in ax[:1]:
    a.annotate("flux ordering", (0.07, 0.42), (0.2, 0.35), arrowprops=dict(arrowstyle="->", color=GREY), fontsize=9)
    a.annotate("Majorana\nfreezing", (1.5, 0.15), (2.5, 0.28), arrowprops=dict(arrowstyle="->", color=GREY), fontsize=9)
save(fig, "kitaev_thermo")

# 5. toric code: KP regions + area law ------------------------------------------
tc, st = toric_code_ground_state(16)
A, B_, C = kitaev_preskill_regions(tc)
fig, ax = plt.subplots(1, 2, figsize=(10, 4.2))
P = np.array([tc.edge_midpoint(e) for e in range(tc.N)]); ax[0].scatter(*P.T, s=4, c="#DDDDDD")
for reg, col, lab in zip((A, B_, C), (RED, TEAL, ORANGE), "ABC"):
    Q = np.array([tc.edge_midpoint(e) for e in reg]); ax[0].scatter(*Q.T, s=14, c=col); ax[0].text(*Q.mean(0), lab, fontsize=16, weight="bold", color="k")
ax[0].set_aspect("equal"); ax[0].axis("off"); ax[0].set_title(r"Kitaev–Preskill:  $\gamma = \ln 2$ exactly")
tcL, stL = toric_code_ground_state(28); c0 = np.array([14.13, 14.13]); S, per = [], []
for R in np.arange(2, 11):
    reg = [e for e in range(tcL.N) if np.linalg.norm(tcL.edge_midpoint(e) - c0) < R]; inside = set(reg)
    S.append(stL.entropy(reg)); per.append(sum(1 for s in tcL.stars() if 0 < len(inside & set(s)) < 4))
per, S = np.array(per), np.array(S); fit = np.polyfit(per, S, 1)
xx = np.linspace(0, per.max() * 1.05, 50)
ax[1].plot(xx, np.polyval(fit, xx), c=ORANGE, lw=1.6); ax[1].plot(per, S, "o", c=NAVY)
ax[1].axhline(0, c=GREY, lw=0.6); ax[1].plot(0, fit[1], "s", c=RED, ms=8)
ax[1].text(3, -0.85, r"intercept $=-\gamma/\ln2=%.3f$" % fit[1], color=RED)
ax[1].set_xlabel(r"boundary length $|\partial A|$"); ax[1].set_ylabel(r"$S_A/\ln 2$"); ax[1].set_ylim(-1.5, None)
save(fig, "toric_tee")

# 6. kagome: S(q) + spinon DOS ----------------------------------------------------
lat = lattice.kagome(2, 3); ed = ED(models.kagome_heisenberg(lat), lat.N); E0, psi = ed.ground_state()
Cm = ed.spin_correlations(psi); Bk = lat.reciprocal()
g = np.linspace(-2.2, 2.2, 151); Q = np.array([a * Bk[0] + b * Bk[1] for a in g for b in g])
Q = Q[np.linalg.norm(Q, axis=1) < 1.9 * np.linalg.norm(Bk[0])]
Sq = ed.structure_factor(psi, lat.positions, Q, C=Cm)
fig, ax = plt.subplots(1, 2, figsize=(11, 4.3))
sc = ax[0].scatter(*Q.T, c=Sq, s=7, cmap=CM); ax[0].set_aspect("equal"); ax[0].axis("off"); plt.colorbar(sc, ax=ax[0], label=r"$S(\mathbf{q})$")
ax[0].set_title("18-site kagome: no Bragg peaks")
big = lattice.kagome(30, 30); om = np.linspace(-3, 3, 600)
for name, fl, c in [("U(1) Dirac [0, π]", {"up": 0, "down": 0, "hexagon": np.pi}, TEAL), ("uniform [0, 0]", {"up": 0, "down": 0, "hexagon": 0}, ORANGE)]:
    ans = SpinonAnsatz.from_flux(big, fl); mu = 0.5 * (ans.eps[ans.n_occ - 1] + ans.eps[ans.n_occ])
    ax[1].plot(om, ans.dos(om + mu, eta=0.025), c=c, label=name)
ax[1].axvline(0, c=GREY, lw=0.6); ax[1].set_xlabel(r"$\omega-\mu$ (spinon units)"); ax[1].set_ylabel("spinon DOS"); ax[1].legend()
ax[1].set_xlim(-2, 2); ax[1].set_ylim(0, 1.0)
ax[1].annotate(r"Dirac: $\rho\propto|\omega|$", (0.12, 0.03), (-1.6, 0.7), arrowprops=dict(arrowstyle="->", color=TEAL), color=TEAL, fontsize=12)
save(fig, "kagome_sq_dos")

# 7. VMC: energies of ansatze vs size --------------------------------------------
res = {}
for name, fl in [("Dirac [0,π]", {"up": 0, "down": 0, "hexagon": np.pi}), ("[π,0]", {"up": np.pi, "down": np.pi, "hexagon": 0})]:
    res[name] = []
    for L in [4, 6, 8]:
        for tw in [(0, 0), (np.pi, 0), (0, np.pi), (np.pi, np.pi)]:
            ans = SpinonAnsatz.from_flux(lattice.kagome(L, L), fl, twist=tw)
            if ans.fermi_gap() > 1e-6: break
        r = GutzwillerVMC(ans).run(n_sweeps=400, n_therm=60, n_chains=2)
        res[name].append((3 * L * L, r["E_per_site"], r["err_per_site"]))
fig, ax = plt.subplots(figsize=(5.4, 3.8))
for (name, d), c in zip(res.items(), (TEAL, ORANGE)):
    d = np.array(d); ax.errorbar(1 / d[:, 0], d[:, 1], d[:, 2], fmt="o-", c=c, label=name, capsize=3)
ax.axhline(-0.42866, ls="--", c=NAVY, lw=1.2); ax.text(0.0005, -0.4278, "Ran et al. (2007)  −0.42866", color=NAVY, fontsize=9)
ax.set_xlabel(r"$1/N$"); ax.set_ylabel(r"$E/NJ$ (Gutzwiller VMC)"); ax.legend(loc="center right"); ax.set_xlim(0, 0.023)
ax.set_xticks([0, 0.005, 0.01, 0.015, 0.02])
save(fig, "kagome_vmc")
print(res)

# 8. dynamics: chain S(q,w) + kagome C(T) -----------------------------------------
L = 20; latc, Hc = models.xxz_chain(L); edc = ED(Hc, L); E0c, p0 = edc.ground_state()
qs = np.array([[2 * np.pi * m / L, 0] for m in range(1, L // 2 + 1)]); omg = np.linspace(0, 3.5, 300)
Sqw = edc.sq_omega(p0, E0c, latc.positions, qs, omg, eta=0.07)
T = np.geomspace(0.02, 5, 70); th = ftlm_thermodynamics(models.kagome_heisenberg(lattice.kagome(2, 3)), 18, T, R=12, M=90)
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
ax[0].pcolormesh(qs[:, 0] / np.pi, omg, Sqw.T, shading="nearest", cmap=CM, vmax=np.percentile(Sqw, 98))
k = np.linspace(0, np.pi, 100)
ax[0].plot(k / np.pi, np.pi / 2 * np.sin(k), "--", c=ORANGE, lw=1.6); ax[0].plot(k / np.pi, np.pi * np.sin(k / 2), ":", c=RED, lw=1.8)
ax[0].set_xlabel(r"$q/\pi$"); ax[0].set_ylabel(r"$\omega/J$"); ax[0].set_title(r"two-spinon continuum, $S^{zz}(q,\omega)$")
ax[1].semilogx(T, th["C"], c=NAVY, label=r"$C/N$"); ax2 = ax[1].twinx(); ax2.semilogx(T, th["S"] / np.log(2), c=TEAL, label=r"$S/N\ln2$")
ax2.spines["right"].set_visible(True); ax2.set_ylabel(r"$S/(N\ln 2)$", color=TEAL); ax[1].set_ylabel(r"$C/N$", color=NAVY)
ax[1].set_xlabel(r"$T/J$"); ax[1].set_title("18-site kagome FTLM: low-$T$ singlet peak")
save(fig, "dynamics_thermo")

# 9. triangular scan + DMRG convergence --------------------------------------------
CL = [[2, 2], [-2, 4]]; P_ = lattice.triangular(1, supercell=CL).high_symmetry_points()
def scan(J2):
    lt = lattice.triangular(1, supercell=CL); e = ED(models.j1j2_triangular(lt, J2=J2), 12); E, ps = e.ground_state(); Cc = e.spin_correlations(ps)
    return [e.structure_factor(ps, lt.positions, P_[x], C=Cc)[0] for x in ("K", "M")]
J2s = np.linspace(0, 0.3, 16); R = np.array(qslab.pmap(scan, J2s))
latd = lattice.triangular(6, 4, bc="cylinder"); H = models.j1j2_triangular(latd, J2=0.1); chis, Es = [8, 16, 32, 64, 128], []
for chi in chis:
    d = DMRG(H, latd.N, chi_max=chi); d.run(6, verbose=False); Es.append(d.energy / latd.N)
fig, ax = plt.subplots(1, 2, figsize=(11, 3.9))
ax[0].axvspan(0.07, 0.15, color=ORANGE, alpha=0.18); ax[0].text(0.08, R.max() * 0.95, "QSL window", color=ORANGE)
ax[0].plot(J2s, R[:, 0], "o-", c=NAVY, label=r"$S(K)$  120°"); ax[0].plot(J2s, R[:, 1], "s-", c=TEAL, label=r"$S(M)$  stripe")
ax[0].set_xlabel(r"$J_2/J_1$"); ax[0].set_ylabel(r"$S(\mathbf{q})$ ($\sqrt{12}\times\sqrt{12}$ ED)"); ax[0].legend()
ax[1].loglog(chis[:-1], (np.array(Es) - Es[-1])[:-1], "o-", c=PURPLE)
ax[1].set_xlabel(r"bond dimension $\chi$"); ax[1].set_ylabel(r"$(E_\chi-E_{128})/N$"); ax[1].set_title(r"DMRG, $6\times4$ YC cylinder, $J_2=0.1$")
save(fig, "triangular")

# 10. NQS convergence ------------------------------------------------------------------
latq = lattice.square(4); fig, ax = plt.subplots(figsize=(5.6, 3.8))
for J2, ls in [(0.0, "-"), (0.5, "--")]:
    Hq = models.j1j2_square(latq, J2=J2); Eex = ED(Hq, 16).ground_state()[0]
    for marsh, c in [(True, TEAL), (False, ORANGE)]:
        q = NQS(Hq, 16, alpha=1, marshall=latq.sublattice_sign() if marsh else None, seed=1); q.optimize(120, lr=0.05, verbose=0)
        ax.semilogy([abs((e - Eex) / Eex) for e, _ in q.history], ls=ls, c=c, label=f"$J_2={J2}$, Marshall={'on' if marsh else 'off'}")
ax.set_xlabel("SR iteration"); ax.set_ylabel("relative energy error"); ax.legend(fontsize=8.5)
save(fig, "nqs")
