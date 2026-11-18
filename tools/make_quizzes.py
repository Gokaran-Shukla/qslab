"""Generates qslab/quizzes/*.json (numbers computed with qslab, see tests)."""
import json, os, math
OUT = os.path.join(os.path.dirname(__file__), "..", "qslab", "quizzes")
os.makedirs(OUT, exist_ok=True)
LN2 = math.log(2)

def mcq(q, opts, ans, expl): return {"type": "mcq", "question": q, "options": opts, "answer": ans, "explanation": expl}
def num(q, ans, tol, expl, hint="", code=""): return {"type": "numeric", "question": q, "answer": ans, "tol": tol, "explanation": expl, "hint": hint, "code": code}

Q = {}
Q["kitaev"] = {"title": "Kitaev honeycomb model", "intro": "H = -sum K_g sigma^g sigma^g on x, y, z bonds (Pauli units unless stated).", "questions": [
 mcq("Which operators commute with the Kitaev Hamiltonian on every hexagon?",
     ["S_tot^z", "W_p = sigma^x sigma^y sigma^z sigma^x sigma^y sigma^z (outward bond labels)", "sum_i sigma^z_i", "S_i . S_j on z bonds"], "b",
     "W_p uses on each vertex the Pauli of the bond pointing out of the hexagon; it commutes with every bond term, so the fluxes are conserved Z2 numbers."),
 mcq("In Kitaev's representation sigma^a_j = i b^a_j c_j, what physical constraint must states satisfy?",
     ["b^x b^y b^z c = 1 on every site (D_j = 1)", "sum_j c_j = 0", "u_jk = +1 on all bonds", "fermion number = N/2"], "a",
     "The Majorana space is 4-dimensional per site; D_j = 1 projects onto the 2 physical states. qslab enforces prod_j D_j = 1 exactly (Pfaffian parity)."),
 mcq("Lieb's theorem implies the ground state of the Kitaev model lies in which flux sector (for large clusters)?",
     ["W_p = -1 on all plaquettes", "W_p = +1 on all plaquettes (flux-free)", "alternating fluxes", "random fluxes"], "b",
     "Flux-free. Tiny clusters can violate this, e.g. 2x2 where the ED ground state has W_p = -1."),
 num("Ground-state energy per site of the isotropic Kitaev model (K = 1, Pauli units) in the thermodynamic limit?", -0.7873, 2e-4,
     "-0.78730 per site (-1.5746 per unit cell).", "integrate -1/2 eps(k) over the BZ",
     "from qslab.kitaev import kitaev_energy_tl\nkitaev_energy_tl(nk=200)"),
 num("Same model written with spin operators, H = K sum S^g S^g (|K| = 1): energy per site?", -0.19682, 3e-4,
     "S = sigma/2 so energies divide by 4: -0.7873/4 = -0.1968.", "", "kitaev_energy_tl()/4"),
 num("Exact ground-state energy per site of the 18-site (3x3 cells) torus, K = 1?", -0.79397, 2e-5,
     "ED and the projected Majorana solution agree: -14.29150/18.",
     "", "km = KitaevMajorana(lattice.honeycomb(3,3)); km.ground_state_energy()/18"),
 num("Nearest-neighbour correlation <sigma^g_i sigma^g_j> on a g-bond (isotropic, TL)?", 0.5249, 2e-3,
     "Hellmann-Feynman: E/N = -(3/2) K <sigma sigma>, so <sigma sigma> = 0.7873/1.5. All further-neighbour spin correlations vanish exactly.",
     "", "KitaevMajorana(lattice.honeycomb(20,20)).bond_correlations()"),
 num("Energy cost of a pair of adjacent visons (flip one u_jk), isotropic K = 1, large cluster?", 0.265, 0.01,
     "~0.26 K: the flux gap is small compared with the Majorana bandwidth (6 K).", "",
     "KitaevMajorana(lattice.honeycomb(20,20)).vison_pair_energy()"),
 mcq("For (Kx, Ky, Kz) = (0.2, 0.2, 1) the system is in", ["the gapless B phase", "a gapped A_z phase equivalent to the toric code", "a Neel state", "a chiral phase"], "b",
     "|Kz| > |Kx| + |Ky| violates the triangle inequality: Majoranas gap out and 4th-order perturbation theory gives the toric code."),
 mcq("Adding the 3-spin term kappa (from a [111] magnetic field) to the B phase gives",
     ["a trivial paramagnet", "a gapped chiral phase with Majorana Chern number +-1 (Ising anyons)", "a ferromagnet", "nothing: kappa commutes trivially"], "b",
     "kappa opens a Majorana gap Delta ~ 6 sqrt(3) kappa; Chern number +-1 means half-quantised thermal Hall kappa_xy/T = (pi/12) k_B^2/hbar."),
 num("Entropy per site on the plateau between the two specific-heat peaks (fraction of ln 2)?", 0.5, 0.05,
     "Itinerant Majoranas freeze out at T_H ~ 1.5 K (Pauli units; 0.375 J in spin units with J = 4K), but the Z2 fluxes only order at T_L ~ 0.05 K (0.012 J): half of ln 2 survives in between.",
     "", "majorana_monte_carlo(L=4)['S']/np.log(2)"),
]}

Q["toric_code"] = {"title": "Toric code / Z2 topological order", "intro": "H = -Je sum_s A_s - Jm sum_p B_p on an L x L torus.", "questions": [
 num("Ground-state energy of the L = 3 toric code (Je = Jm = 1)?", -18, 1e-6, "-(Je + Jm) L^2 = -18; every stabiliser is satisfied.", "", "ED(*reversed(models.toric_code(3))...)"),
 num("Ground-state degeneracy on the torus?", 4, 0, "4 = |anyon types|: Wilson loops along both cycles take values +-1."),
 num("Ground-state degeneracy on a genus-2 surface?", 16, 0, "4^g for a Z2 gauge theory."),
 num("Energy of the lowest excitation (Je = Jm = 1)?", 4, 1e-6, "Anyons come in pairs; each violated stabiliser costs 2, so the lowest e-pair costs 4."),
 num("Topological entanglement entropy gamma (in nats)?", 0.6931, 1e-3, "gamma = ln D with total quantum dimension D = 2 for Z2: ln 2.",
     "", "tc, st = stabilizer.toric_code_ground_state(12); stabilizer.topological_entropy(st, *stabilizer.kitaev_preskill_regions(tc))"),
 mcq("Why does the Kitaev-Preskill combination S_A+S_B+S_C-S_AB-S_BC-S_CA+S_ABC isolate -gamma?",
     ["Boundary (area-law) and corner contributions cancel by inclusion-exclusion", "It is the mutual information", "It vanishes for all states", "It counts Bell pairs"], "a",
     "Every boundary segment appears with alternating signs; only the topological constant survives."),
 mcq("Which is the e-particle (charge) of the toric code?", ["violated plaquette B_p = -1", "violated star A_s = -1", "a Majorana fermion", "a magnon"], "b",
     "Charges are star violations created by Z strings; fluxes (m) are plaquette violations created by X strings on the dual lattice. e x m = fermion."),
 mcq("A large uniform field h_z eventually", ["confines e particles (Higgs/confinement transition)", "does nothing", "makes the code chiral", "doubles the degeneracy"], "a",
     "The field gives anyons kinetic energy; they condense near h ~ 0.33 (3D-Ising* transition). Try it with ED on L = 3 by watching the splitting of the 4 states."),
]}

Q["kagome"] = {"title": "Kagome Heisenberg antiferromagnet", "intro": "Corner-sharing triangles, 3 sites per unit cell.", "questions": [
 num("Exact ground-state energy per site of the 12-site (2x2 cell) kagome torus?", -0.45374, 2e-5, "Small cluster; the TL value is about -0.4386 (DMRG).", "",
     "lat = lattice.kagome(2,2); ED(models.kagome_heisenberg(lat), 12).ground_state()[0]/12"),
 mcq("Which spinon flux pattern defines the U(1) Dirac spin liquid (Ran-Hermele-Lee-Wen)?",
     ["0 through triangles, pi through hexagons", "pi through triangles, 0 through hexagons", "pi/2 through triangles", "0 everywhere"], "a",
     "[0, pi]: Dirac cones at the Fermi level; Gutzwiller energy ~ -0.4287 J."),
 num("VMC energy per site of the projected U(1) Dirac spin liquid (large torus)?", -0.4287, 2e-3,
     "Known value -0.42866(1); qslab gives -0.4291(9) on 108 sites.", "",
     "ans = SpinonAnsatz.from_flux(lattice.kagome(6,6), {'up':0,'down':0,'hexagon':np.pi}); GutzwillerVMC(ans).run()"),
 mcq("What is the classical ground-state degeneracy of the kagome AFM?", ["unique 120-degree state", "extensive (any configuration with total spin 0 on every triangle)", "two-fold", "Neel"], "b",
     "Each triangle only needs S_triangle = 0: extensive degeneracy, so quantum fluctuations select the state."),
 mcq("Which of these stabilises a chiral spin liquid on kagome?", ["J2 = J3 ~ 0.5 or a scalar chirality term Jchi", "an Ising anisotropy", "a ferromagnetic J1", "a staggered field"], "a",
     "He-Sheng-Chen (2014), Gong et al. (2014): a Kalmeyer-Laughlin state with C = 1/2."),
 num("Topological entanglement entropy expected for a gapped Z2 spin liquid on kagome (nats)?", 0.6931, 1e-3, "Same anyon theory as the toric code: gamma = ln 2 (DMRG finds about 0.69)."),
 mcq("Why must the spinon Fermi sea be a closed shell before Gutzwiller projection?",
     ["Otherwise the projected state depends on arbitrary choices in a degenerate level", "Projection is impossible", "Because of SU(2)", "It need not be"], "a",
     "Use twisted (antiperiodic) spinon boundary conditions or another cluster; qslab warns about open shells."),
]}

Q["triangular"] = {"title": "Triangular J1-J2 Heisenberg model", "intro": "", "questions": [
 mcq("Ground state of the triangular J1 Heisenberg model?", ["120-degree Neel order", "spin liquid", "VBS", "ferromagnet"], "a",
     "Three-sublattice 120-degree order with a strongly reduced moment, about 0.2."),
 mcq("Roughly where is the J2/J1 spin-liquid window?", ["0.07 - 0.15", "0.5 - 0.6", "1 - 2", "none"], "a",
     "DMRG (Zhu-White, Hu et al. 2015) and VMC; recent work favours the U(1) Dirac spin liquid."),
 num("ED energy per site on the 12-site (3x4) torus at J2 = 0?", -0.57757, 2e-5, "", "",
     "lat = lattice.triangular(3,4); ED(models.j1j2_triangular(lat, J2=0), 12).ground_state()[0]/12"),
 num("Same cluster at J2 = 0.1?", -0.54206, 2e-5, "J2 frustrates the 120-degree state and raises the energy."),
 mcq("Which momentum peaks in S(q) for 120-degree order?", ["K (Brillouin-zone corner)", "Gamma", "M", "none"], "a",
     "Three-sublattice order is a K-point Bragg peak; stripe order (large J2) peaks at M."),
 mcq("Why do DMRG studies use YC/XC cylinders?", ["the MPS entanglement grows with circumference, and the orientation selects which momenta are allowed", "they are faster than tori for no reason", "SU(2) requires cylinders", "to break time reversal"], "a",
     "Area law: S grows linearly with width W, so the cost grows as exp(W)."),
]}

Q["square_j1j2"] = {"title": "Square J1-J2 model and benchmarks", "intro": "", "questions": [
 num("ED ground-state energy per site of the 4x4 Heisenberg AFM (J2 = 0)?", -0.70178, 1e-5, "", "",
     "ED(models.heisenberg(lattice.square(4)), 16).ground_state()[0]/16"),
 num("Same at the maximally frustrated point J2 = 0.5?", -0.52862, 1e-5, ""),
 num("Exact energy per site of the Gutzwiller-projected pi-flux state on 4x4 (antiperiodic spinon BCs)?", -0.63773, 2e-4,
     "Its overlap with the true ground state is 0.85.", "",
     "SpinonAnsatz.from_flux(lattice.square(4), {'square': np.pi}, twist=(0, np.pi)) + projected_state"),
 mcq("Why does an RBM need the Marshall sign on the square lattice but struggles on kagome?",
     ["The bipartite AFM ground state has a known sign rule; frustrated lattices do not", "RBMs cannot be complex", "Because of Sz conservation", "No reason"], "a",
     "Marshall: sign = (-1)^(number of down spins on sublattice A). No such rule exists for frustrated magnets."),
 mcq("Which phase is believed to lie at 0.45 < J2 < 0.6?", ["gapless spin liquid and/or plaquette VBS (debated)", "ferromagnet", "120-degree order", "toric code"], "a", "Still actively debated."),
]}

Q["kitaev_heisenberg_gamma"] = {"title": "Generalised Kitaev materials (J-K-Gamma-Gamma')", "intro": "alpha-RuCl3, Na2IrO3, Co-based honeycombs.", "questions": [
 mcq("What microscopic mechanism produces bond-dependent Kitaev exchange?",
     ["Strong spin-orbit coupling, j_eff = 1/2, edge-sharing octahedra with 90-degree M-X-M bonds", "Dipolar coupling", "Direct exchange only", "Weak SOC"], "a",
     "Jackeli-Khaliullin: the two exchange paths cancel the Heisenberg part and leave K."),
 mcq("Sign of K in alpha-RuCl3 (spin units, H = K S S)?", ["ferromagnetic, K < 0", "antiferromagnetic, K > 0", "zero", "it depends on the field direction"], "a", "K ~ -5 to -10 meV, with Gamma > 0."),
 num("ED energy per site of the 12-site cluster with (J, K, Gamma, Gamma', J3) = (-0.1, -1, 0.5, -0.025, 0.1)?", -0.38088, 2e-5, "", "",
     "models.kitaev_heisenberg_gamma(lattice.honeycomb(2,3), J=-0.1, K=-1, Gamma=0.5, Gamma_p=-0.025, J3=0.1)"),
 mcq("What does the zero-field ground state of alpha-RuCl3 show?", ["zigzag order (T_N ~ 7 K)", "a spin liquid", "Neel order", "ferromagnetism"], "a",
     "Zigzag; a field of about 7-8 T in-plane suppresses it, and the resulting state is debated (proximate Kitaev spin liquid?)."),
 mcq("Which DMRG requirement does the Gamma term change?", ["it makes the MPO complex because of the S^x S^y terms", "it breaks the MPO", "nothing", "it requires 3-site terms"], "a",
     "qslab writes sigma^y as -i times a real matrix; with an odd number of y's per term the MPO is complex."),
 num("Ground-state energy per site of the pure FM Kitaev model in spin units (K = -1) on 12 sites?", -0.20417, 2e-5,
     "FM and AFM Kitaev models are related by a unitary sublattice rotation, so their spectra are identical."),
]}

Q["methods"] = {"title": "Methods and diagnostics for spin liquids", "intro": "ED, DMRG, partons/VMC, NQS, thermodynamics, parallel computing", "questions": [
 mcq("Why can't ARPACK reliably count a 4-fold topological degeneracy with a tiny Krylov space?",
     ["Lanczos generates a single vector per degenerate eigenspace from one start vector; restarts or a large Krylov space are needed", "It is complex", "Because of round-off", "It always can"], "a",
     "qslab uses ncv >= 40; symmetry operators then label the sectors."),
 mcq("Which quantity distinguishes a gapped Z2 spin liquid from a trivial paramagnet?", ["topological entanglement entropy gamma = ln 2", "zero magnetisation", "finite gap", "<S_i.S_j> < 0"], "a",
     "Area law S = alpha L - gamma; only topological order has gamma > 0."),
 mcq("A smoking gun for fractionalisation in neutron scattering is", ["a broad continuum in S(q, w) instead of sharp magnons", "a Bragg peak", "a spin gap", "a Curie law"], "a", "Spinons are created in pairs, so the two-particle states form a continuum."),
 mcq("The QMC sign problem prevents", ["unbiased QMC for frustrated Heisenberg models", "ED", "DMRG", "Majorana MC for Kitaev"], "a",
     "The Kitaev model is sign-free in the Majorana representation, so Majorana MC is exact up to statistics."),
 mcq("In qslab FTLM, independent random vectors are distributed over", ["threads or MPI ranks via pmap", "GPUs only", "they cannot be parallelised", "BLAS"], "a", "Embarrassingly parallel: R random vectors and all Sz sectors."),
 mcq("Which parallel strategy suits a single huge ED matrix-vector product on one node?",
     ["shared-memory Numba 'pull' kernel (each thread owns output rows)", "MPI over independent tasks", "running more Lanczos vectors", "a process pool"], "a",
     "The pull formulation needs no atomics; scipy CSR matvec is single-threaded."),
 num("How many parameters does an RBM with alpha = 2 have for N = 16 spins?", 560, 0, "N + M + N M with M = 32: 16 + 32 + 512 = 560."),
 mcq("DMRG entanglement on a cylinder of width W grows as", ["S ~ W (area law), so chi ~ exp(W)", "S ~ log L", "S constant", "S ~ W^2"], "a", "That is why 2D DMRG is limited to W of about 10-12."),
]}

import random
_rng = random.Random(2026)
for k, v in Q.items():
    for q in v["questions"]:
        if q["type"] == "mcq":          # shuffle so the answer letter is not predictable
            idx = "abcdefgh".index(q["answer"])
            correct = q["options"][idx]
            _rng.shuffle(q["options"])
            q["answer"] = "abcdefgh"[q["options"].index(correct)]
    with open(os.path.join(OUT, f"{k}.json"), "w") as f:
        json.dump(v, f, indent=1)
print("wrote", list(Q))
