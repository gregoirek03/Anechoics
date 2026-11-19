# -*- coding: utf-8 -*-


# Python packages
import matplotlib.pyplot
import numpy
import numpy as np          # CORRECTION 0 : import en haut du fichier (avant toute utilisation de np)
import os
import compute_alpha          # matériau : LS26 semi-infini (compute_alpha_ls26)


# MRG packages
import _env
import preprocessing
import processing
import postprocessing
#import solutions

c0 = 3e8                    # CORRECTION 0 : constantes globales définies une seule fois, en haut
mu0 = 4 * np.pi * 1e-7


def derive_J_proj(domain_omega, spacestep, chi, Alpha, u, p):
    # CORRECTION 1 : le gradient est une FONCTION sur le mur (une valeur par nœud),
    # pas un nombre multiplié par chi0. En chaque nœud b du mur :
    #     dJ/dchi_b = Re(Alpha * u_b * p_b) * h
    # Le solveur ne garde pas u sur le mur : on le reconstruit depuis le nœud
    # intérieur voisin par u_b = u_voisin / (1 - Alpha*chi_b*h).
    (M, N) = numpy.shape(domain_omega)
    h = spacestep
    j_prime = numpy.zeros((M, N))
    for i in range(1, M - 1):
        for j in range(1, N - 1):
            if domain_omega[i, j] != _env.NODE_INTERIOR:
                continue
            for (ib, jb) in ((i - 1, j), (i + 1, j), (i, j - 1), (i, j + 1)):
                if domain_omega[ib, jb] == _env.NODE_ROBIN:
                    c = 1.0 / (1.0 - Alpha * chi[ib, jb] * h)
                    j_prime[ib, jb] += np.real(Alpha * (c * u[i, j]) * (c * p[i, j])) * h
    return j_prime


def projection(chi_tilde, domain_omega, V_obj):
    # CORRECTION 2 (nouvelle fonction) : après le pas de gradient, chi doit rester
    # entre 0 et 1 ET garder la même quantité de matériau (somme = V_obj * S).
    # On cherche l tel que somme(clip(chi_tilde + l, 0, 1)) = V_obj * S, par dichotomie.
    mur = (domain_omega == _env.NODE_ROBIN)
    S = numpy.sum(mur)
    x = chi_tilde[mur]
    l_min, l_max = -1.0 - x.max(), 1.0 - x.min()
    for _ in range(100):
        l = (l_min + l_max) / 2
        if numpy.clip(x + l, 0, 1).sum() > V_obj * S:
            l_max = l
        else:
            l_min = l
    chi = numpy.zeros_like(chi_tilde)
    chi[mur] = numpy.clip(x + l, 0, 1)
    return chi


def binarisation(chi, domain_omega, V_obj):
    # chi doit valoir 0 ou 1 : on met du matériau (chi = 1) sur les n_mat nœuds du mur
    # où le chi relaxé (optimisé dans [0, 1]) est le plus grand, avec n_mat = V_obj * S
    # pour garder exactement le même volume de matériau.
    mur = (domain_omega == _env.NODE_ROBIN)
    idx = numpy.argwhere(mur)
    n_mat = int(round(V_obj * len(idx)))
    ordre = numpy.argsort(-chi[mur], kind='stable')      # du plus grand au plus petit
    chi_bin = numpy.zeros_like(chi)
    for (i, j) in idx[ordre[:n_mat]]:
        chi_bin[i, j] = 1.0
    return chi_bin


def binarisation_tramage(chi, domain_omega):
    """Binarisation par diffusion d'erreur le long du mur (motif alterné 0/1).
    L'optimum multi-fréquence est une densité INTERMÉDIAIRE (chi ~ 0.2 à 0.7 partout) :
    garder les 40 % plus grands nœuds donne un bloc arbitraire qui recrée des résonances.
    Ici on parcourt le mur selon x en cumulant chi ; on pose du matériau (chi = 1) dès que
    le cumul dépasse 1/2 : la densité locale de matériau suit celle du chi relaxé et le
    volume total est conservé (à un nœud près)."""
    mur = (domain_omega == _env.NODE_ROBIN)
    idx = numpy.argwhere(mur)
    idx = idx[numpy.argsort(idx[:, 1], kind='stable')]
    chi_bin = numpy.zeros_like(chi)
    cumul = 0.0
    for (i, j) in idx:
        cumul += chi[i, j]
        if cumul >= 0.5:
            chi_bin[i, j] = 1.0
            cumul -= 1.0
    return chi_bin


def indicatrice_interet(domain_omega):
    # CORRECTION 3 : la chambre occupe les lignes 0 à N du tableau (M = 2N), pas 0 à M.
    # L'ancienne zone M//3 : 2M//3 (lignes 33 à 66) traversait le mur (ligne 50).
    (M, N) = numpy.shape(domain_omega)
    ind = numpy.zeros((M, N))                 # zeros((M, N)) avec un tuple, pas zeros(M, N)
    ind[N // 3: 2 * N // 3, N // 3: 2 * N // 3] = 1.0   # divisions entières //
    return ind


def your_compute_objective_function(domain_omega, u, spacestep):

    (M, N) = numpy.shape(domain_omega)

    # 1. Énergie sur tout le domaine de calcul (u vaut 0 hors de la chambre)
    energy_total = numpy.sum(numpy.abs(u)**2) * (spacestep**2)

    # 2. Énergie uniquement sur la zone d'intérêt (CORRECTION 3)
    energy_interest = numpy.sum(indicatrice_interet(domain_omega) * numpy.abs(u)**2) * (spacestep**2)

    # 3. Fonction J globale
    energy = energy_total + energy_interest

    return energy


def your_optimization_procedure(domain_omega, spacestep, omega, f, f_dir, f_neu, f_rob,
                           beta_pde, alpha_pde, alpha_dir, beta_neu, beta_rob, alpha_rob,
                           Alpha, mu, chi, V_obj):
    """This function return the optimized density.

    Parameter:
        cf solvehelmholtz's remarks
        Alpha: complex, it corresponds to the absorbtion coefficient;
        mu: float, it is the initial step of the gradient's descent;
        V_obj: float, it characterizes the volume constraint on the density chi.
    """
    # CORRECTION 4 : dans ce code, "omega" EST déjà le nombre d'onde k (sans unité) :
    # c'est ce que reçoit solve_helmholtz. Pas de omega // c0.
    wavenumber = omega

    k = 0
    (M, N) = numpy.shape(domain_omega)
    w = 1 + indicatrice_interet(domain_omega)          # poids de J : 1 partout, 2 dans la zone
    numb_iter = 100
    energy = numpy.zeros((numb_iter+1, 1), dtype=numpy.float64)
    zeros = numpy.zeros((M, N), dtype=numpy.complex128)

    while k < numb_iter and mu > 10**(-5):
        pass  # print('1. computing solution of Helmholtz problem, i.e., u')
        alpha_rob = Alpha * chi
        u = processing.solve_helmholtz(domain_omega, spacestep, wavenumber, f, f_dir, f_neu, f_rob,
                                       beta_pde, alpha_pde, alpha_dir, beta_neu, beta_rob, alpha_rob)

        pass  # print('2. computing solution of adjoint problem, i.e., p')
        # CORRECTION 5 : second membre -2 * w * conj(u) (sans le facteur omega)
        # et p = 0 sur la source : on passe des zéros à la place de f_dir.
        f_adjoint = -2 * w * np.conj(u)
        p = processing.solve_helmholtz(domain_omega, spacestep, wavenumber, f_adjoint, zeros, f_neu, f_rob,
                                       beta_pde, alpha_pde, alpha_dir, beta_neu, beta_rob, alpha_rob)

        pass  # print('3. computing objective function, i.e., energy')
        energy[k] = your_compute_objective_function(domain_omega, u, spacestep)
        ene = energy[k]

        pass  # print('4. computing parametric gradient')
        j_prime = derive_J_proj(domain_omega, spacestep, chi, Alpha, u, p)
        chi_old = chi.copy()    # CORRECTION 6 : on garde le chi de départ de l'itération

        while ene >= energy[k] and mu > 10 ** -5:
            pass  # print('    a. computing gradient descent')
            chi = chi_old - mu * j_prime          # toujours repartir de chi_old
            pass  # print('    b. computing projected gradient')
            chi = projection(chi, domain_omega, V_obj)          # CORRECTION 2
            pass  # print('    c. computing solution of Helmholtz problem, i.e., u')
            u = processing.solve_helmholtz(domain_omega, spacestep, wavenumber, f, f_dir, f_neu, f_rob,
                                           beta_pde, alpha_pde, alpha_dir, beta_neu, beta_rob, Alpha * chi)
            pass  # print('    d. computing objective function, i.e., energy (E)')
            ene = your_compute_objective_function(domain_omega, u, spacestep)   # CORRECTION 7 : bon nom
            if ene < energy[k]:
                # The step is increased if the energy decreased
                mu = mu * 1.1
            else:
                # The step is decreased is the energy increased
                mu = mu / 2
        if ene >= energy[k]:     # aucun pas n'a fait baisser J : on garde l'ancien chi et on s'arrête
            chi = chi_old
            break
        print('itération %3d   J = %.5f   (pas mu = %.3g)' % (k, float(ene), mu))
        k += 1

    energy[k] = your_compute_objective_function(domain_omega, u, spacestep)
    print('end. computing solution of Helmholtz problem, i.e., u')

    return chi, energy[:k+1], u, j_prime        # CORRECTION 8 : "grad" n'existait pas


def norme_p(Js, p):
    """Max « lissé » des énergies sur la bande : (somme J_i^p)^(1/p), proche de max J_i
    pour p grand, mais dérivable (le vrai max ne l'est pas quand deux pics s'échangent)."""
    Jmax = Js.max()
    return Jmax * numpy.sum((Js / Jmax) ** p) ** (1.0 / p)


def your_optimization_procedure_multi(domain_omega, spacestep, ks, f, f_dir, f_neu, f_rob,
                                      beta_pde, alpha_pde, alpha_dir, beta_neu, beta_rob,
                                      chi, V_obj, p=10, mu=0.2, numb_iter=5):
    """Optimisation de chi sur une BANDE de fréquences : on minimise le max (lissé, norme p)
    de J(chi, k) sur les k de `ks`, avec Alpha(k) du LS26 à chaque k.

    Gradient : d(J_p) = somme_i (J_i / J_p)^(p-1) dJ_i, chaque dJ_i par la méthode adjointe
    (derive_J_proj). Seuls les k qui pèsent dans le max (poids > 1e-4) demandent un adjoint.
    Le pas mu est en unités de chi : on avance de mu * gradient / max|gradient|.
    Renvoie chi (relaxé, dans [0,1]) et l'historique [J_p, J_max, J_moyen] par itération."""
    (M, N) = numpy.shape(domain_omega)
    mur = (domain_omega == _env.NODE_ROBIN)
    w = 1 + indicatrice_interet(domain_omega)
    zeros = numpy.zeros((M, N), dtype=numpy.complex128)
    Alphas = [compute_alpha.compute_alpha_ls26(k) for k in ks]

    def evaluer(chi):
        us, Js = [], numpy.zeros(len(ks))
        for i, (k, A) in enumerate(zip(ks, Alphas)):
            u = processing.solve_helmholtz(domain_omega, spacestep, k, f, f_dir, f_neu, f_rob,
                                           beta_pde, alpha_pde, alpha_dir, beta_neu, beta_rob, A * chi)
            us.append(u)
            Js[i] = your_compute_objective_function(domain_omega, u, spacestep)
        return Js, us

    Js, us = evaluer(chi)
    Jp = norme_p(Js, p)
    historique = [[Jp, Js.max(), Js.mean()]]
    print('itération   0   J_p = %.3f   J_max = %.3f (k = %.2f)   J_moyen = %.3f'
          % (Jp, Js.max(), ks[numpy.argmax(Js)], Js.mean()))

    for it in range(1, numb_iter + 1):
        # -- gradient de J_p
        poids = (Js / Jp) ** (p - 1)
        grad = numpy.zeros((M, N))
        for i in numpy.flatnonzero(poids > 1e-4 * poids.max()):
            padj = processing.solve_helmholtz(domain_omega, spacestep, ks[i], -2 * w * np.conj(us[i]), zeros,
                                              f_neu, f_rob, beta_pde, alpha_pde, alpha_dir, beta_neu,
                                              beta_rob, Alphas[i] * chi)
            grad += poids[i] * derive_J_proj(domain_omega, spacestep, chi, Alphas[i], us[i], padj)
        gmax = numpy.abs(grad[mur]).max()
        if gmax == 0:
            break
        grad /= gmax

        # -- recherche du pas (même règle que la version mono-fréquence)
        chi_old = chi
        accepte = False
        while mu > 1e-4:
            chi_new = projection(chi_old - mu * grad, domain_omega, V_obj)
            Js_new, us_new = evaluer(chi_new)
            Jp_new = norme_p(Js_new, p)
            if Jp_new < Jp:
                chi, Js, us, Jp = chi_new, Js_new, us_new, Jp_new
                mu *= 1.2
                accepte = True
                break
            mu /= 2
        if not accepte:          # aucun pas ne fait baisser J_p : on s'arrête
            break
        historique.append([Jp, Js.max(), Js.mean()])
        print('itération %3d   J_p = %.3f   J_max = %.3f (k = %.2f)   J_moyen = %.3f   (pas mu = %.3g)'
              % (it, Jp, Js.max(), ks[numpy.argmax(Js)], Js.mean(), mu))

    return chi, numpy.array(historique)


def compute_alpha_semi_infini(omega, epsilon_r, sigma):
    """
    Calcule le coefficient d'absorption complexe alpha(omega)
    pour un matériau donné (matériau semi-infini, incidence normale).
    """
    # Équation pour le nombre d'onde complexe dans le milieu
    terme_reel = epsilon_r / (c0**2)
    terme_imaginaire = (sigma * mu0) / omega

    # CORRECTION 9 : le code utilise la convention exp(+j omega t) (Sommerfeld = -1j*k).
    # Le nombre d'onde du matériau s'écrit alors k_m = omega*sqrt(eps/c0^2 - j sigma mu0/omega),
    # avec un signe MOINS devant j.
    km = omega * np.sqrt(terme_reel - 1j * terme_imaginaire)

    # Le coefficient alpha(omega) pour la condition de Robin
    alpha = -1j * km

    return alpha

# CORRECTION 10 : compute_alpha_mag était une copie exacte de compute_alpha (sans mu_r).
# Pour la ferrite, utiliser compute_alpha.py (empilement ferrite + mousse).


def plot_resultats(domain_omega, u0, un, chi0, chin, energy, filename='fig_resultats.png', show=True):
    """Figure de synthèse : |u| avant/après, chi le long du mur, historique de J."""
    import matplotlib.patches as patches
    plt = matplotlib.pyplot
    (M, N) = numpy.shape(domain_omega)
    mur = (domain_omega == _env.NODE_ROBIN)
    rows = slice(0, N + 1)                       # on ne montre que la chambre (+ le mur)

    a0, an = numpy.abs(u0[rows]), numpy.abs(un[rows])
    vmax = max(a0.max(), an.max())

    fig = plt.figure(figsize=(14, 8))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.3, 1])

    # -- 1. |u| avant / après / différence (même échelle pour comparer)
    champs = [(a0, r'$|u_0|$ (avant optimisation)', 'viridis', 0, vmax),
              (an, r'$|u_n|$ (après optimisation)', 'viridis', 0, vmax),
              (an - a0, r'$|u_n| - |u_0|$', 'RdBu_r', -numpy.abs(an - a0).max(), numpy.abs(an - a0).max())]
    for c, (data, titre, cmap, vmin, vm) in enumerate(champs):
        ax = fig.add_subplot(gs[0, c])
        im = ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vm, extent=(0, 1, (N + 1) / N, 0))
        ax.add_patch(patches.Rectangle(((N // 3) / N, (N // 3) / N), (N // 3) / N, (N // 3) / N,
                                       fill=False, ec='white' if c < 2 else 'k', ls='--', lw=1.5))
        ax.set_title(titre)
        ax.set_xlabel('x')
        ax.set_ylabel('y (source en haut, mur en bas)')
        fig.colorbar(im, ax=ax, shrink=0.8)

    # -- 2. Répartition du matériau le long du mur
    ax = fig.add_subplot(gs[1, 0:2])
    xs = numpy.arange(N) / N
    def profil(chi):
        return numpy.array([chi[mur[:, j], j].mean() if mur[:, j].any() else numpy.nan for j in range(N)])
    ax.step(xs, profil(chi0), where='mid', color='gray', ls='--', label=r'$\chi_0$ (initial)')
    ax.fill_between(xs, profil(chin), step='mid', alpha=0.35, color='C3')
    ax.step(xs, profil(chin), where='mid', color='C3', lw=2, label=r'$\chi_n$ (optimisé)')
    ax.axvspan((N // 3) / N, (2 * N // 3) / N, color='C0', alpha=0.08, label="abscisses de la zone d'intérêt")
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel('position x le long du mur')
    ax.set_ylabel(r'densité $\chi$')
    ax.set_title('Répartition du matériau absorbant (volume conservé : %.3f → %.3f)'
                 % (chi0[mur].mean(), chin[mur].mean()))
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(alpha=0.3)

    # -- 3. Historique de la fonctionnelle J
    ax = fig.add_subplot(gs[1, 2])
    e = numpy.ravel(energy)
    ax.plot(numpy.arange(len(e)), e, 'o-', ms=3, color='C0')
    ax.set_xlabel('itération')
    ax.set_ylabel('J(χ)')
    ax.set_title('Décroissance de J : %.4f → %.4f (%.1f %%)' % (e[0], e[-1], 100 * (e[-1] / e[0] - 1)))
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(filename, dpi=150)
    print('Figure de synthèse enregistrée :', filename)
    if show:
        plt.show()      # sinon la figure reste ouverte et s'affichera au plt.show() final
    return fig


def mur_total_absorbant(plage_wavenumbers=None, k_choisi=None, chi_mur=None, verbose=True, mur_initial=False):
    """Balayage en fréquence, mur plan 100 % absorbant (ou répartition chi_mur si fournie,
    ou mur initial du sujet à 40 % si mur_initial=True).
    Renvoie (plage_wavenumbers, energies). k_choisi : k marqué sur la figure."""
    # -- Paramètres de base
    N = 50
    M = 2 * N
    level = 0  # level=0 garantit une paroi plane (plane shape)
    spacestep = 1.0 / N

    # -- Préparation des variables pour stocker les résultats
    if plage_wavenumbers is None:
        plage_wavenumbers = numpy.linspace(1.0, 20.0, 40) # Teste 40 fréquences entre 1 et 20
    liste_energies = []

    # -- Initialisation de la géométrie et des matrices de base (inchangé)
    beta_pde, alpha_pde, alpha_dir, beta_neu, alpha_rob, beta_rob = preprocessing._set_coefficients_of_pde(M, N)
    f, f_dir, f_neu, f_rob = preprocessing._set_rhs_of_pde(M, N)
    domain_omega, x, y, _, _ = preprocessing._set_geometry_of_domain(M, N, level)

    # -- Forcer le mur à être TOTALEMENT absorbant
    chi = numpy.zeros((M, N), dtype=numpy.float64)
    chi[domain_omega == _env.NODE_ROBIN] = 1.0
    if chi_mur is not None:       # répartition imposée (ex. chi avant / après optimisation)
        chi = chi_mur
    elif mur_initial:             # répartition initiale du sujet (_set_chi : 40 % du mur)
        chi = preprocessing.set2zero(preprocessing._set_chi(M, N, x, y), domain_omega)

    # -- Définition de l'onde incidente (source plane)
    f_dir[:, :] = 0.0
    f_dir[0, 0:N] = 1.0

    # -- BOUCLE SUR LES FRÉQUENCES
    # Ici 1 unité du code = 1 m : wavenumber est en m^-1 et omega = k * c0 en rad/s.
    if verbose:
        print("Début du balayage en fréquence...")
    for wavenumber in plage_wavenumbers:
        omega = wavenumber * c0
        Alpha = compute_alpha.compute_alpha_ls26(wavenumber)   # LS26 semi-infini (eps 9.6, tan delta 1.14)
        alpha_rob = Alpha * chi
        u = processing.solve_helmholtz(domain_omega, spacestep, wavenumber, f, f_dir, f_neu, f_rob,
                            beta_pde, alpha_pde, alpha_dir, beta_neu, beta_rob, alpha_rob)
        energie_actuelle = your_compute_objective_function(domain_omega, u, spacestep)
        liste_energies.append(energie_actuelle)
        if verbose:
            print(f"Wavenumber: {wavenumber:.2f} | Énergie: {energie_actuelle:.4f}")

    if k_choisi is None:          # appel simple : on renvoie les données sans tracer
        return numpy.array(plage_wavenumbers), numpy.array(liste_energies)

    # -- Tracer la courbe pour identifier les maxima locaux
    matplotlib.pyplot.figure()
    matplotlib.pyplot.plot(plage_wavenumbers, liste_energies, marker='o')
    matplotlib.pyplot.axvline(k_choisi, color='C3', ls='--',
                              label='k optimisé = %.3f (énergie max)' % k_choisi)
    matplotlib.pyplot.legend()
    matplotlib.pyplot.title("Énergie électromagnétique selon k (mur plan 100 % absorbant)")
    matplotlib.pyplot.xlabel("Wavenumber (k)")
    matplotlib.pyplot.ylabel("Énergie totale J")
    matplotlib.pyplot.grid(True)
    matplotlib.pyplot.savefig('fig_balayage_frequence.jpg')
    # on ne ferme pas la figure : elle s'affichera au plt.show() final
    return numpy.array(plage_wavenumbers), numpy.array(liste_energies)


def plot_perte_binarisation(domain_omega, ks, E_relaxe, E_bin, chi_relaxe, chi_bin,
                            filename='fig_perte_binarisation.png', fenetre=5):
    """Perte due à la binarisation : J(k) sur la bande pour chi relaxé (dans [0,1]) et
    chi binaire (0/1, motif alterné), et profil de chi le long du mur.
    La densité locale du binaire (moyenne glissante sur `fenetre` nœuds) montre que le
    motif alterné suit bien la densité du chi relaxé."""
    plt = matplotlib.pyplot
    (M, N) = numpy.shape(domain_omega)
    mur = (domain_omega == _env.NODE_ROBIN)
    xs = numpy.arange(N) / N

    def profil(chi):
        return numpy.array([chi[mur[:, j], j].mean() if mur[:, j].any() else numpy.nan for j in range(N)])

    p_rel, p_bin = profil(chi_relaxe), profil(chi_bin)
    densite_bin = numpy.convolve(numpy.nan_to_num(p_bin), numpy.ones(fenetre) / fenetre, mode='same')

    fig, axs = plt.subplots(1, 2, figsize=(14, 5.2), gridspec_kw={'width_ratios': [1.6, 1]})

    # -- 1. J(k) sur la bande, relaxé vs binaire
    ax = axs[0]
    ax.plot(ks, E_relaxe, color='C2', lw=2,
            label=r'$\chi$ relaxé : J_max = %.2f, J moyen = %.2f' % (E_relaxe.max(), E_relaxe.mean()))
    ax.plot(ks, E_bin, color='C3', lw=2,
            label=r'$\chi$ binaire : J_max = %.2f, J moyen = %.2f' % (E_bin.max(), E_bin.mean()))
    ax.fill_between(ks, E_relaxe, E_bin, where=E_bin > E_relaxe, color='C3', alpha=0.15,
                    label='perte due à la binarisation')
    ax.set_yscale('log')
    ax.set_xlabel("nombre d'onde k")
    ax.set_ylabel('énergie J (échelle log)')
    ax.set_title('Perte de binarisation : J_max x %.1f, J moyen x %.2f'
                 % (E_bin.max() / E_relaxe.max(), E_bin.mean() / E_relaxe.mean()))
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    # -- 2. chi le long du mur
    ax = axs[1]
    ax.fill_between(xs, p_bin, step='mid', color='C3', alpha=0.25)
    ax.step(xs, p_bin, where='mid', color='C3', lw=1, label=r'$\chi$ binaire (motif alterné)')
    ax.plot(xs, p_rel, color='C2', lw=2.5, label=r'$\chi$ relaxé')
    ax.plot(xs, densite_bin, color='k', ls='--', lw=1.5,
            label='densité locale du binaire (moyenne sur %d nœuds)' % fenetre)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel('position x le long du mur')
    ax.set_ylabel(r'densité $\chi$')
    ax.set_title('Volume : relaxé %.3f / binaire %.3f' % (chi_relaxe[mur].mean(), chi_bin[mur].mean()))
    ax.legend(fontsize=8, loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=1)
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(filename, dpi=150)
    print('Figure enregistrée :', filename)
    return fig


def plot_balayage_avant_apres(k_opt, chi0, chin, filename='fig_balayage_avant_apres.png', nom_opt=None):
    """J(k) sur k dans [1, 20] pour : mur plein, chi0 (avant) et chin (optimisé à k_opt).
    Même matériau (LS26 semi-infini, eps_r = 9.6, tan delta = 1.14) à chaque k."""
    plt = matplotlib.pyplot
    ks = numpy.union1d(numpy.round(numpy.arange(1.0, 20.0001, 0.1), 6),
                       numpy.linspace(k_opt - 0.3, k_opt + 0.3, 31))      # plus fin autour du pic
    print('Balayage en fréquence avant / après optimisation (%d valeurs de k x 3)...' % len(ks))
    (M, N) = numpy.shape(chi0)          # mur plat : S = N nœuds
    courbes = [('mur plein (100 % du mur)', None, '0.55', ':'),
               (r'$\chi_0$ avant optimisation (%.0f %% du mur)' % (100 * chi0[chi0 > 0].size / N), chi0, 'C0', '-'),
               (nom_opt or r'$\chi_n$ optimisé à k = %.3f (binaire, même volume)' % k_opt, chin, 'C3', '-')]
    res = []
    for nom, chi, couleur, ls in courbes:
        _, E = mur_total_absorbant(ks, chi_mur=chi, verbose=False)
        res.append(E)
        i_opt = numpy.argmin(numpy.abs(ks - k_opt))
        print('  %-55s J(k_opt) = %7.3f | J moyen sur [1, 20] = %.3f | J max = %.3f (k = %.2f)'
              % (nom.replace('$', ''), E[i_opt], E.mean(), E.max(), ks[numpy.argmax(E)]))

    fig, axs = plt.subplots(1, 2, figsize=(14, 5.2), gridspec_kw={'width_ratios': [2.2, 1]})
    for ax, (xmin, xmax) in zip(axs, [(1, 20), (k_opt - 1.0, k_opt + 1.0)]):
        for (nom, _, couleur, ls), E in zip(courbes, res):
            sel = (ks >= xmin) & (ks <= xmax)
            ax.plot(ks[sel], E[sel], ls=ls, color=couleur, lw=2 if couleur != '0.55' else 1.5, label=nom)
        ax.axvline(k_opt, color='k', ls='--', lw=1)
        ax.set_xlim(xmin, xmax)
        ax.set_xlabel('nombre d\'onde k')
        ax.set_ylabel('énergie J')
        ax.grid(alpha=0.3)
    axs[0].set_yscale('log')            # pics de résonance très hauts : échelle log pour tout voir
    axs[0].set_ylabel('énergie J (échelle log)')
    axs[0].set_title('Énergie selon k : avant / après optimisation de la répartition')
    axs[1].set_title('Zoom autour de k = %.3f (k optimisé)' % k_opt)
    axs[0].legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(filename, dpi=150)
    print('Figure enregistrée :', filename)
    return fig


if __name__ == '__main__':

    # ----------------------------------------------------------------------
    # -- Fell free to modify the function call in this cell.
    # ----------------------------------------------------------------------
    # -- set parameters of the geometry
    N = 50  # number of points along x-axis
    M = 2 * N  # number of points along y-axis
    level = 0 # level of the fractal
    spacestep = 1.0 / N  # mesh size

    # -- set parameters of the partial differential equation
    kx = -1.0
    ky = -1.0
    wavenumber = numpy.sqrt(kx**2 + ky**2)  # wavenumber
    wavenumber = 10.0

    # -- OPTIMISER_AU_MAX = True : on cherche d'abord la fréquence où l'énergie est la plus
    #    haute AVEC LE MUR INITIAL (chi0 de _set_chi, 40 % du mur couvert) : balayage sur
    #    k dans [1, 20] au pas 0.1 (pics étroits), puis balayage fin autour du pic.
    #    En mono-fréquence, on optimise chi à CETTE fréquence, avec l'Alpha du matériau à cette
    #    fréquence. En multi-fréquence (OPTIMISER_MULTI_FREQ), ce k ne sert qu'aux figures.
    #    OPTIMISER_AU_MAX = False : valeurs du sujet (k = 10, Alpha = 10 - 10j).
    OPTIMISER_AU_MAX = True
    if OPTIMISER_AU_MAX:
        ks, Es = mur_total_absorbant(numpy.round(numpy.arange(1.0, 20.0001, 0.1), 6),
                                     mur_initial=True, verbose=False)   # balayage grossier
        k_grossier = ks[numpy.argmax(Es)]
        pas = ks[1] - ks[0]
        ks_fin, Es_fin = mur_total_absorbant(numpy.linspace(k_grossier - pas, k_grossier + pas, 21),
                                             mur_initial=True, verbose=False)
        k_max = ks_fin[numpy.argmax(Es_fin)]
        ks_tout = numpy.concatenate([ks, ks_fin])
        Es_tout = numpy.concatenate([Es, Es_fin])
        ordre = numpy.argsort(ks_tout)
        wavenumber = k_max
        print('==> mur initial (40 %%) : énergie maximale J = %.2f à k = %.3f (f = %.0f MHz si 1 unité = 1 m)'
              % (Es_fin.max(), k_max, k_max * c0 / (2 * np.pi) / 1e6))
    # ----------------------------------------------------------------------
    # -- Do not modify this cell, these are the values that you will be assessed against.
    # ----------------------------------------------------------------------
    # --- set coefficients of the partial differential equation
    beta_pde, alpha_pde, alpha_dir, beta_neu, alpha_rob, beta_rob = preprocessing._set_coefficients_of_pde(M, N)

    # -- set right hand sides of the partial differential equation
    f, f_dir, f_neu, f_rob = preprocessing._set_rhs_of_pde(M, N)

    # -- set geometry of domain
    domain_omega, x, y, _, _ = preprocessing._set_geometry_of_domain(M, N, level)

    # ----------------------------------------------------------------------
    # -- Fell free to modify the function call in this cell.
    # ----------------------------------------------------------------------
    # -- define boundary conditions
    # planar wave defined on top
    f_dir[:, :] = 0.0
    f_dir[0, 0:N] = 1.0
    # spherical wave defined on top
    #f_dir[:, :] = 0.0
    #f_dir[0, int(N/2)] = 10.0

    # -- initialize
    alpha_rob[:, :] = - wavenumber * 1j

    # -- define material density matrix
    chi = preprocessing._set_chi(M, N, x, y)
    chi = preprocessing.set2zero(chi, domain_omega)

    # -- define absorbing material
    Alpha = 10.0 - 10.0 * 1j
    if OPTIMISER_AU_MAX:      # même matériau que dans le balayage, évalué au k du pic
        Alpha = compute_alpha.compute_alpha_ls26(wavenumber)
        print('Alpha au pic =', Alpha)
    # -- this is the function you have written during your project
    # Notre absorbant hybride (ferrite Ni-Zn + mousse carbone), cf. compute_alpha.py.
    # USE_OUR_MATERIAL = False : valeurs du sujet (k = 10, Alpha = 10 - 10j), celles de l'évaluation.
    # USE_OUR_MATERIAL = True  : k et Alpha calculés pour la fréquence F_HZ, 1 unité du code = L_REF mètres
    #                            (garder N >= 20 * F_HZ * L_REF / 3e8).
    USE_OUR_MATERIAL = False
    F_HZ, L_REF = 100e6, 3.0
    if USE_OUR_MATERIAL:
        import compute_alpha
        wavenumber, Alpha = compute_alpha.compute_alpha(
            F_HZ, L_ref=L_REF, d_ferrite=5.5e-3, K=1999.0, fm=3.94e6, eps_ferrite=14.2,
            h_foam=0.30, foam_eps_p=1.5, foam_sigma=0.003)
        print('f =', F_HZ / 1e6, 'MHz : wavenumber =', wavenumber, ', Alpha =', Alpha)
    alpha_rob = Alpha * chi

    # -- set parameters for optimization
    S = 0  # surface of the fractal
    for i in range(0, M):
        for j in range(0, N):
            if domain_omega[i, j] == _env.NODE_ROBIN:
                S += 1
    V_0 = 1  # initial volume of the domain
    V_obj = numpy.sum(numpy.sum(chi)) / S  # constraint on the density
    mu = 5  # initial gradient step
    mu1 = 10**(-5)  # parameter of the volume functional

    # ----------------------------------------------------------------------
    # -- Do not modify this cell, these are the values that you will be assessed against.
    # ----------------------------------------------------------------------
    # -- compute finite difference solution
    u = processing.solve_helmholtz(domain_omega, spacestep, wavenumber, f, f_dir, f_neu, f_rob,
                        beta_pde, alpha_pde, alpha_dir, beta_neu, beta_rob, alpha_rob)
    chi0 = chi.copy()
    u0 = u.copy()

    # ----------------------------------------------------------------------
    # -- Fell free to modify the function call in this cell.
    # ----------------------------------------------------------------------
    # -- compute optimization
    # CORRECTION 11 : l'appel était commenté, l'optimisation ne tournait jamais.
    # OPTIMISER_MULTI_FREQ = True : on minimise le MAX de l'énergie sur toute la bande
    # k dans [1, 20] (pas 0.1), au lieu de J à un seul k. energy = historique de J_max.
    # OPTIMISER_MULTI_FREQ = False : optimisation mono-fréquence au k choisi plus haut.
    OPTIMISER_MULTI_FREQ = True
    if OPTIMISER_MULTI_FREQ:
        ks_bande = numpy.round(numpy.arange(1.0, 20.0001, 0.1), 6)
        print('Optimisation multi-fréquence : max de J sur %d valeurs de k dans [%.1f, %.1f]'
              % (len(ks_bande), ks_bande[0], ks_bande[-1]))
        chi, historique = your_optimization_procedure_multi(domain_omega, spacestep, ks_bande, f, f_dir, f_neu, f_rob,
                                                            beta_pde, alpha_pde, alpha_dir, beta_neu, beta_rob,
                                                            chi, V_obj)
        energy = historique[:, 1:2]                         # J_max à chaque itération
        J_relaxe = float(energy[-1, 0])
        chi_relaxe = chi.copy()                             # gardé pour la figure de perte
        chi = binarisation_tramage(chi, domain_omega)       # motif alterné (cf. docstring)
        _, Es_relaxe = mur_total_absorbant(ks_bande, chi_mur=chi_relaxe, verbose=False)
        _, Es_bin = mur_total_absorbant(ks_bande, chi_mur=chi, verbose=False)
        J_binaire = Es_bin.max()
        print('J_max relaxé = %.3f | J_max binaire = %.3f (k = %.2f) | J_moyen binaire = %.3f'
              % (J_relaxe, J_binaire, ks_bande[numpy.argmax(Es_bin)], Es_bin.mean()))
        plot_perte_binarisation(domain_omega, ks_bande, Es_relaxe, Es_bin, chi_relaxe, chi)
    else:
        chi, energy, u, grad = your_optimization_procedure(domain_omega, spacestep, wavenumber, f, f_dir, f_neu, f_rob,
                            beta_pde, alpha_pde, alpha_dir, beta_neu, beta_rob, alpha_rob,
                            Alpha, mu, chi, V_obj)
        J_relaxe = float(energy[-1, 0])
        chi = binarisation(chi, domain_omega, V_obj)
    # --- en of optimization

    # -- chi ne peut valoir que 0 ou 1 : binarisation du chi relaxé, puis nouveau calcul de u
    # (en multi-fréquence, u est le champ au k du pic initial, pour les figures)
    u = processing.solve_helmholtz(domain_omega, spacestep, wavenumber, f, f_dir, f_neu, f_rob,
                                   beta_pde, alpha_pde, alpha_dir, beta_neu, beta_rob, Alpha * chi)
    if not OPTIMISER_MULTI_FREQ:
        J_binaire = your_compute_objective_function(domain_omega, u, spacestep)
        print('J relaxé (chi dans [0,1]) = %.5f | J binaire (chi dans {0,1}) = %.5f' % (J_relaxe, J_binaire))
    energy = numpy.vstack([energy, [[J_binaire]]])     # dernier point = chi binaire

    chin = chi.copy()
    un = u.copy()

    # -- plot chi, u, and energy
    postprocessing._plot_uncontroled_solution(u0, chi0)
    postprocessing._plot_controled_solution(un, chin)
    err = un - u0
    postprocessing._plot_error(err)
    postprocessing._plot_energy_history(energy)
    print('J initial =', float(energy[0, 0]), '| J final =', float(energy[-1, 0]),
          '| gain = %.1f %%' % (100 * (energy[-1, 0] / energy[0, 0] - 1)))
    plot_resultats(domain_omega, u0, un, chi0, chin, energy, show=False)

    # CORRECTION 12 : le balayage en fréquence est appelé ICI, et non tout en bas
    # du fichier hors du "if __name__" (sinon il tourne aussi à chaque import).
    if OPTIMISER_AU_MAX:      # figure du balayage (déjà calculé) avec le k choisi marqué
        matplotlib.pyplot.figure()
        matplotlib.pyplot.plot(ks_tout[ordre], Es_tout[ordre], marker='o', ms=3)
        matplotlib.pyplot.axvline(k_max, color='C3', ls='--', label='k optimisé = %.3f (énergie max)' % k_max)
        matplotlib.pyplot.title("Énergie selon k (mur initial, 40 % couvert)")
        matplotlib.pyplot.xlabel("Wavenumber (k)")
        matplotlib.pyplot.ylabel("Énergie totale J")
        matplotlib.pyplot.legend()
        matplotlib.pyplot.grid(True)
        matplotlib.pyplot.savefig('fig_balayage_frequence.jpg')
        plot_balayage_avant_apres(k_max, chi0, chin, nom_opt=(r'$\chi_n$ optimisé sur le max de J, k dans [1, 20] (binaire, motif alterné)'
                                                              if OPTIMISER_MULTI_FREQ else None))
    else:
        mur_total_absorbant(k_choisi=wavenumber)

    print('End.')
    # -- affichage à l'écran de toutes les figures restées ouvertes
    # (synthèse + balayage en fréquence) ; le script se termine quand on les ferme.
    matplotlib.pyplot.show()