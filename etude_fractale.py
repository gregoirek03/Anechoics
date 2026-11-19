# -*- coding: utf-8 -*-
"""Étude des murs fractals (motif de Koch quadratique) à quantité de matériau ÉGALE.

Pour chaque niveau de fractale (0, 1, 2) et chaque quantité de matériau V_abs (nombre
de nœuds absorbants, même valeur absolue pour tous les niveaux), on optimise chi par
gradient projeté depuis 3 départs, on garde le meilleur, on binarise, et on compare.

Réutilise les fonctions de demo_control_polycopie2024.py (gradient, projection, J,
binarisation) sans les dupliquer. processing.py n'est pas modifié.
"""

# Python packages
import csv
import os
import time

import matplotlib.pyplot as plt
import numpy

# MRG packages
import _env
import preprocessing
import processing
import demo_control_polycopie2024 as demo


# ----------------------------------------------------------------------------
# Paramètres de l'étude
# ----------------------------------------------------------------------------
N = 64
NIVEAUX = [0, 1, 2]
# 48 retiré : avec 5 valeurs le calcul dépassait ~15 min (~22 s par optimisation, 45 optimisations)
LISTE_V_ABS = [8, 16, 24, 32]
V_ABS_FIGURE = 32          # quantité de matériau pour les figures par niveau (50 % du mur plat)
V_ABS_ECONOMIE = 32        # référence mur plat pour le calcul d'économie de matériau
WAVENUMBER = 10.0          # valeurs du sujet
ALPHA = 10.0 - 10.0 * 1j
MU0 = 5.0                  # pas initial (comme dans le demo)
MU_MIN = 1e-5
MAX_ITER = 60              # nombre max d'itérations de gradient par optimisation
TOL_REL = 1e-4             # arrêt si J baisse de moins de 0,01 % sur 5 itérations
N_VERIF = 128              # maillage fin pour la vérification
GRAINE = 0                 # graine du départ aléatoire (reproductible)

DOSSIER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'figures_fractales')


# ----------------------------------------------------------------------------
# Construction d'un cas (géométrie + coefficients)
# ----------------------------------------------------------------------------
def construire_cas(N, level):
    """Géométrie et coefficients du problème pour un niveau de fractale."""
    if N % 4 ** level != 0:
        raise ValueError('N = %d doit être un multiple de 4^level = %d' % (N, 4 ** level))
    creneau = N // 4 ** level
    if level > 0 and creneau < 4:
        raise ValueError('plus petit créneau = %d points (< 4) : augmenter N' % creneau)
    M = 2 * N
    beta_pde, alpha_pde, alpha_dir, beta_neu, alpha_rob, beta_rob = preprocessing._set_coefficients_of_pde(M, N)
    f, f_dir, f_neu, f_rob = preprocessing._set_rhs_of_pde(M, N)
    domain_omega, x, y, _, _ = preprocessing._set_geometry_of_domain(M, N, level)
    f_dir[:, :] = 0.0
    f_dir[0, 0:N] = 1.0                                    # onde plane
    mur = (domain_omega == _env.NODE_ROBIN)
    return dict(N=N, M=M, level=level, h=1.0 / N, domain=domain_omega, x=x, y=y, mur=mur,
                S=int(mur.sum()), creneau=creneau,
                f=f, f_dir=f_dir, f_neu=f_neu, f_rob=f_rob, beta_pde=beta_pde, alpha_pde=alpha_pde,
                alpha_dir=alpha_dir, beta_neu=beta_neu, beta_rob=beta_rob)


def resoudre(cas, chi, f=None, f_dir=None):
    """u (ou p si f et f_dir sont fournis) pour la densité chi."""
    c = cas
    return processing.solve_helmholtz(c['domain'], c['h'], WAVENUMBER,
                                      c['f'] if f is None else f,
                                      c['f_dir'] if f_dir is None else f_dir,
                                      c['f_neu'], c['f_rob'], c['beta_pde'], c['alpha_pde'],
                                      c['alpha_dir'], c['beta_neu'], c['beta_rob'], ALPHA * chi)


def calcul_J(cas, chi):
    u = resoudre(cas, chi)
    return demo.your_compute_objective_function(cas['domain'], u, cas['h']), u


# ----------------------------------------------------------------------------
# Variante de la projection et de l'optimisation : contrainte en volume ABSOLU
# ----------------------------------------------------------------------------
def projection_volume_absolu(chi_tilde, domain_omega, V_abs):
    """Projette chi sur {0 <= chi <= 1, somme(chi sur le mur) = V_abs}.
    La projection du demo impose somme = V_obj * S : on lui passe V_obj = V_abs / S."""
    S = int(numpy.sum(domain_omega == _env.NODE_ROBIN))
    if not 0 < V_abs <= S:
        raise ValueError('V_abs = %g hors de ]0, S = %d]' % (V_abs, S))
    return demo.projection(chi_tilde, domain_omega, V_abs / S)


def optimisation_volume_absolu(cas, chi_depart, V_abs):
    """Gradient projeté (même schéma que your_optimization_procedure du demo :
    adjoint, derive_J_proj, pas adaptatif x1.1 / ÷2), avec la contrainte en volume absolu.
    Différences : u accepté réutilisé (une résolution de moins par itération) et arrêt
    quand J stagne. Renvoie chi, historique de J, u, nombre d'itérations."""
    dom, h = cas['domain'], cas['h']
    w = 1 + demo.indicatrice_interet(dom)                    # poids de J (cf. demo)
    zeros = numpy.zeros_like(cas['f'])
    chi = projection_volume_absolu(chi_depart, dom, V_abs)
    J, u = calcul_J(cas, chi)
    historique = [J]
    mu = MU0
    for it in range(MAX_ITER):
        p = resoudre(cas, chi, f=-2 * w * numpy.conj(u), f_dir=zeros)   # adjoint (cf. demo)
        grad = demo.derive_J_proj(dom, h, chi, ALPHA, u, p)
        accepte = False
        while mu > MU_MIN:
            chi_essai = projection_volume_absolu(chi - mu * grad, dom, V_abs)
            J_essai, u_essai = calcul_J(cas, chi_essai)
            if J_essai < J:
                chi, J, u = chi_essai, J_essai, u_essai
                mu *= 1.1
                accepte = True
                break
            mu /= 2
        if not accepte:
            break
        historique.append(J)
        if len(historique) > 5 and historique[-6] - J < TOL_REL * historique[-6]:
            break
    return chi, numpy.array(historique), u, len(historique) - 1


def departs(cas, V_abs, rng):
    """Les 3 points de départ : uniforme, aléatoire projeté, _set_chi remis à l'échelle."""
    dom, mur, S = cas['domain'], cas['mur'], cas['S']
    uniforme = numpy.zeros((cas['M'], cas['N']))
    uniforme[mur] = V_abs / S
    aleatoire = numpy.zeros_like(uniforme)
    aleatoire[mur] = rng.random(S)
    aleatoire = projection_volume_absolu(aleatoire, dom, V_abs)
    sujet = preprocessing.set2zero(preprocessing._set_chi(cas['M'], cas['N'], cas['x'], cas['y']), dom)
    sujet = projection_volume_absolu(sujet * V_abs / sujet.sum(), dom, V_abs)
    return [('uniforme', uniforme), ('aleatoire', aleatoire), ('set_chi', sujet)]


# ----------------------------------------------------------------------------
# Figures
# ----------------------------------------------------------------------------
def figure_J_vs_V(resultats, J_plein):
    """J_plein : référence unique = mur PLAT entièrement couvert (niveau 0, chi = 1 partout)."""
    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.axhline(J_plein, color='k', ls=':', lw=1.5,
               label='référence : mur plat plein (χ = 1 sur les %d nœuds) — J = %.3f' % (N, J_plein))
    for c, lv in enumerate(NIVEAUX):
        lignes = [r for r in resultats if r['niveau'] == lv]
        V = [r['V_abs'] for r in lignes]
        ax.plot(V, [r['J_opt_binaire'] for r in lignes], 'o-', color='C%d' % c, lw=2,
                label='niveau %d — χ binaire 0/1' % lv)
        ax.plot(V, [r['J_opt_continu'] for r in lignes], '--', color='C%d' % c, lw=1, alpha=0.5,
                label='niveau %d — χ continu (avant binarisation)' % lv)
    ax.set_xlabel('quantité de matériau V_abs (nombre de nœuds absorbants)')
    ax.set_ylabel('J optimal')
    ax.set_title('J optimal selon la quantité de matériau, à matériau égal\n'
                 '(k = %g, Alpha = %s, N = %d)' % (WAVENUMBER, ALPHA, N))
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, ncol=1)
    fig.tight_layout()
    fig.savefig(os.path.join(DOSSIER, 'J_opt_vs_V_abs.png'), dpi=150)
    return fig


def figure_niveau(cas, chi_avant, u_avant, J_avant, chi_opt, u_opt, J_opt, J_continu, V_abs):
    """chi_opt, u_opt, J_opt : chi BINAIRE final et la solution correspondante."""
    lv, Nn = cas['level'], cas['N']
    mur = cas['mur']
    i_max = int(numpy.argwhere(mur)[:, 0].max()) + 2           # on coupe sous le mur
    a0, a1 = numpy.abs(u_avant[:i_max]), numpy.abs(u_opt[:i_max])
    vmax = max(a0.max(), a1.max())
    iw, jw = numpy.nonzero(mur)

    fig, axs = plt.subplots(1, 3, figsize=(15, 6.2))
    for ax, a, titre in [(axs[0], a0, '|u| avant : χ uniforme = V_abs/S\nJ = %.4f' % J_avant),
                         (axs[1], a1, '|u| après optimisation, χ binaire\nJ = %.4f (χ continu : %.4f)'
                          % (J_opt, J_continu))]:
        im = ax.imshow(a, cmap='viridis', vmin=0, vmax=vmax, origin='upper',
                       extent=(-0.5, Nn - 0.5, i_max - 0.5, -0.5))
        ax.plot(cas['x'], cas['y'], color='w', lw=0.8)
        ax.set_xlim(-0.5, Nn - 0.5)
        ax.set_title(titre)
        ax.set_xlabel('j (colonne)')
        ax.set_ylabel('i (ligne) — source en haut')
        fig.colorbar(im, ax=ax, shrink=0.75, label='|u|')

    ax = axs[2]
    ax.plot(cas['x'], cas['y'], color='0.7', lw=1, zorder=1)
    plein = chi_opt[mur] == 1
    ax.scatter(jw[~plein], iw[~plein], c='0.92', s=18, edgecolors='0.6', linewidths=0.3, zorder=2,
               label='χ = 0 (mur nu)')
    ax.scatter(jw[plein], iw[plein], c='C3', s=30, edgecolors='k', linewidths=0.4, zorder=3,
               label='χ = 1 (matériau)')
    ax.set_xlim(-1, Nn)
    i_haut, i_bas = iw.min(), iw.max()
    marge = max(4, (16 - (i_bas - i_haut)) / 2)      # mur plat : on garde une hauteur lisible
    ax.set_ylim(i_bas + marge, i_haut - marge)
    ax.set_aspect('equal')
    ax.set_title('χ binaire sur les nœuds du mur\n(S = %d nœuds, %d nœuds à 1)' % (cas['S'], int(chi_opt[mur].sum())))
    ax.set_xlabel('j (colonne)')
    ax.set_ylabel('i (ligne)')
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 0.0), borderaxespad=4.0, ncol=2, fontsize=8,
              frameon=False)

    fig.suptitle('Mur fractal niveau %d — V_abs = %d nœuds absorbants (plus petit créneau : %d points)'
                 % (lv, V_abs, cas['creneau']))
    fig.tight_layout()
    fig.savefig(os.path.join(DOSSIER, 'niveau_%d_V%d.png' % (lv, V_abs)), dpi=150)
    return fig


# ----------------------------------------------------------------------------
# Vérification de maillage
# ----------------------------------------------------------------------------
def reechantillonner_chi(cas_grossier, chi, cas_fin):
    """chi sur le mur fin = chi du nœud de mur grossier le plus proche (en coordonnées
    physiques). Le maillage fin a 2x plus de nœuds par unité de longueur et h deux fois
    plus petit : la quantité physique de matériau (somme chi * h) est conservée."""
    pg = numpy.argwhere(cas_grossier['mur']) * cas_grossier['h']
    vg = chi[cas_grossier['mur']]
    idx_fin = numpy.argwhere(cas_fin['mur'])
    pf = idx_fin * cas_fin['h']
    d2 = ((pf[:, None, :] - pg[None, :, :]) ** 2).sum(axis=2)
    chi_fin = numpy.zeros((cas_fin['M'], cas_fin['N']))
    chi_fin[tuple(idx_fin.T)] = vg[d2.argmin(axis=1)]
    return chi_fin


def V_equivalent(V, J, J_cible):
    """Plus petite quantité V (interpolation linéaire) pour laquelle J(V) <= J_cible."""
    if J[0] <= J_cible:
        return None, '<= %d' % V[0]
    for a in range(len(V) - 1):
        if J[a + 1] <= J_cible:
            t = (J[a] - J_cible) / (J[a] - J[a + 1])
            v = V[a] + t * (V[a + 1] - V[a])
            return v, '%.1f' % v
    return None, '> %d (non atteint)' % V[-1]


# ----------------------------------------------------------------------------
# Programme principal
# ----------------------------------------------------------------------------
if __name__ == '__main__':
    os.makedirs(DOSSIER, exist_ok=True)
    t_debut = time.time()
    rng = numpy.random.default_rng(GRAINE)
    resultats, cas_par_niveau, meilleurs = [], {}, {}
    figures = []

    # -- référence UNIQUE : mur plat (niveau 0) entièrement couvert, chi = 1 sur ses N nœuds.
    #    (un fractal entièrement couvert aurait 2x ou 4x plus de matériau : pas comparable)
    cas_plat = construire_cas(N, 0)
    plein = numpy.zeros((cas_plat['M'], N))
    plein[cas_plat['mur']] = 1.0
    J_plein, _ = calcul_J(cas_plat, plein)
    print('Référence mur plat plein (χ = 1 sur %d nœuds) : J = %.4f' % (cas_plat['S'], J_plein))

    print('Étude fractale : N = %d, k = %g, Alpha = %s, V_abs = %s' % (N, WAVENUMBER, ALPHA, LISTE_V_ABS))
    for lv in NIVEAUX:
        cas = cas_plat if lv == 0 else construire_cas(N, lv)
        cas_par_niveau[lv] = cas
        print('\n--- niveau %d : S = %d nœuds de mur, plus petit créneau = %d points'
              % (lv, cas['S'], cas['creneau']))

        for V_abs in LISTE_V_ABS:
            if V_abs > cas['S']:
                continue
            t_V = time.time()
            meilleur = None
            for nom, chi0 in departs(cas, V_abs, rng):
                t0 = time.time()
                chi, hist, u, n_it = optimisation_volume_absolu(cas, chi0, V_abs)
                print('  niveau %d | V_abs = %2d | départ %-9s | J : %.4f -> %.4f | %2d itérations | %5.1f s'
                      % (lv, V_abs, nom, hist[0], hist[-1], n_it, time.time() - t0))
                if meilleur is None or hist[-1] < meilleur['J']:
                    meilleur = dict(J=hist[-1], chi=chi, u=u, n_it=n_it, depart=nom)
            chi_bin = demo.binarisation(meilleur['chi'], cas['domain'], V_abs / cas['S'])
            J_bin, _ = calcul_J(cas, chi_bin)
            meilleurs[(lv, V_abs)] = meilleur
            resultats.append(dict(niveau=lv, S=cas['S'], V_abs=V_abs, J_mur_plat_plein=J_plein,
                                  J_opt_continu=meilleur['J'], J_opt_binaire=J_bin,
                                  iterations=meilleur['n_it'], depart_retenu=meilleur['depart'],
                                  temps_s=round(time.time() - t_V, 1)))

    # -- CSV
    chemin_csv = os.path.join(DOSSIER, 'resultats_fractales.csv')
    with open(chemin_csv, 'w', newline='') as fcsv:
        ecrivain = csv.DictWriter(fcsv, fieldnames=list(resultats[0].keys()))
        ecrivain.writeheader()
        ecrivain.writerows(resultats)

    # -- figures
    figures.append(figure_J_vs_V(resultats, J_plein))
    for lv in NIVEAUX:
        if (lv, V_ABS_FIGURE) not in meilleurs:
            continue
        cas, m = cas_par_niveau[lv], meilleurs[(lv, V_ABS_FIGURE)]
        chi_unif = numpy.zeros((cas['M'], N))
        chi_unif[cas['mur']] = V_ABS_FIGURE / cas['S']
        J_unif, u_unif = calcul_J(cas, chi_unif)
        chi_bin = demo.binarisation(m['chi'], cas['domain'], V_ABS_FIGURE / cas['S'])
        J_bin, u_bin = calcul_J(cas, chi_bin)
        figures.append(figure_niveau(cas, chi_unif, u_unif, J_unif, chi_bin, u_bin, J_bin, m['J'], V_ABS_FIGURE))

    # -- vérification de maillage sur le meilleur cas
    best = min(resultats, key=lambda r: r['J_opt_continu'])
    lv_b, V_b = best['niveau'], best['V_abs']
    print('\nVérification de maillage : meilleur cas = niveau %d, V_abs = %d (N = %d -> %d)...'
          % (lv_b, V_b, N, N_VERIF))
    cas_fin = construire_cas(N_VERIF, lv_b)
    chi_fin = reechantillonner_chi(cas_par_niveau[lv_b], meilleurs[(lv_b, V_b)]['chi'], cas_fin)
    J_fin, _ = calcul_J(cas_fin, chi_fin)
    ecart = abs(J_fin - best['J_opt_continu']) / best['J_opt_continu']
    longueur_g = meilleurs[(lv_b, V_b)]['chi'].sum() / N
    longueur_f = chi_fin.sum() / N_VERIF

    # -- résumé
    print('\n' + '=' * 78)
    print('RÉSUMÉ (k = %g, Alpha = %s, N = %d) — fichiers dans %s' % (WAVENUMBER, ALPHA, N, DOSSIER))
    print('=' * 78)
    print('\nJ optimal (χ continu) par niveau et par quantité de matériau V_abs :')
    print('  (référence mur plat plein, χ = 1 sur %d nœuds : J = %.4f)' % (cas_par_niveau[0]['S'], J_plein))
    print('  %-9s %5s %12s' % ('niveau', 'S', '') + ''.join('  V=%-6d' % v for v in LISTE_V_ABS))
    for lv in NIVEAUX:
        lignes = {r['V_abs']: r for r in resultats if r['niveau'] == lv}
        print('  %-9d %5d %12s' % (lv, cas_par_niveau[lv]['S'], '')
              + ''.join('  %-8.4f' % lignes[v]['J_opt_continu'] if v in lignes else '  %-8s' % '-'
                        for v in LISTE_V_ABS))
    print('\nJ avec χ binaire 0/1 :')
    for lv in NIVEAUX:
        lignes = {r['V_abs']: r for r in resultats if r['niveau'] == lv}
        print('  %-9d %5s %12s' % (lv, '', '')
              + ''.join('  %-8.4f' % lignes[v]['J_opt_binaire'] if v in lignes else '  %-8s' % '-'
                        for v in LISTE_V_ABS))

    moyenne = {lv: numpy.mean([r['J_opt_continu'] for r in resultats if r['niveau'] == lv]) for lv in NIVEAUX}
    lv_eff = min(moyenne, key=moyenne.get)
    print('\nNiveau le plus efficace (J optimal moyen sur les V_abs) : niveau %d (J moyen = %.4f ; %s)'
          % (lv_eff, moyenne[lv_eff], ', '.join('niv. %d : %.4f' % (lv, moyenne[lv]) for lv in NIVEAUX)))

    J_cible = [r['J_opt_continu'] for r in resultats if r['niveau'] == 0 and r['V_abs'] == V_ABS_ECONOMIE][0]
    print('\nÉconomie de matériau pour atteindre le J du mur plat à V_abs = %d (J = %.4f) :'
          % (V_ABS_ECONOMIE, J_cible))
    for lv in NIVEAUX[1:]:
        lignes = sorted([r for r in resultats if r['niveau'] == lv], key=lambda r: r['V_abs'])
        v, texte = V_equivalent([r['V_abs'] for r in lignes], [r['J_opt_continu'] for r in lignes], J_cible)
        if v is not None:
            print('  niveau %d : V_abs ≈ %s nœuds suffisent -> économie ≈ %.0f %% de matériau'
                  % (lv, texte, 100 * (1 - v / V_ABS_ECONOMIE)))
        elif texte.startswith('<='):
            print('  niveau %d : J cible déjà atteint avec V_abs %s -> économie >= %.0f %%'
                  % (lv, texte, 100 * (1 - LISTE_V_ABS[0] / V_ABS_ECONOMIE)))
        else:
            print('  niveau %d : J cible non atteint dans la plage étudiée (V_abs %s)' % (lv, texte))
    print('  (V_abs compte des nœuds : 1 nœud = h = 1/%d de longueur de mur couverte, même valeur à tous les niveaux)' % N)

    # Si J du mur plat n'est pas décroissant en V_abs, la cible "mur plat à V_abs = 32" est
    # facile à battre : on compare aussi au MEILLEUR mur plat, toutes quantités confondues.
    plat = sorted([r for r in resultats if r['niveau'] == 0], key=lambda r: r['V_abs'])
    J_plat = [r['J_opt_continu'] for r in plat]
    if any(J_plat[a + 1] > J_plat[a] for a in range(len(J_plat) - 1)):
        print('  ATTENTION : pour le mur plat, J optimal n\'est pas décroissant en V_abs (%s) :'
              % ', '.join('%.4f' % j for j in J_plat))
        print('  à V_abs = %d il fait moins bien qu\'avec moins de matériau (contrainte d\'égalité sur le volume).'
              % V_ABS_ECONOMIE)
    r_best = min(plat, key=lambda r: r['J_opt_continu'])
    print('  Comparaison au meilleur mur plat (V_abs = %d, J = %.4f) :' % (r_best['V_abs'], r_best['J_opt_continu']))
    for lv in NIVEAUX[1:]:
        lignes = sorted([r for r in resultats if r['niveau'] == lv], key=lambda r: r['V_abs'])
        v, texte = V_equivalent([r['V_abs'] for r in lignes], [r['J_opt_continu'] for r in lignes],
                                r_best['J_opt_continu'])
        if v is not None:
            print('    niveau %d : atteint ce J avec V_abs ≈ %s -> économie ≈ %.0f %% (vs %d nœuds)'
                  % (lv, texte, 100 * (1 - v / r_best['V_abs']), r_best['V_abs']))
        else:
            print('    niveau %d : V_abs %s' % (lv, texte))

    print('\nVérification de maillage (niveau %d, V_abs = %d, même χ rééchantillonné le long du mur) :'
          % (lv_b, V_b))
    print('  J(N = %d) = %.4f | J(N = %d) = %.4f | écart relatif = %.1f %%'
          % (N, best['J_opt_continu'], N_VERIF, J_fin, 100 * ecart))
    print('  longueur de mur couverte : %.4f (N = %d) vs %.4f (N = %d)' % (longueur_g, N, longueur_f, N_VERIF))
    print('\nTemps total : %.1f min' % ((time.time() - t_debut) / 60))
    print('CSV : %s' % chemin_csv)

    plt.show()
