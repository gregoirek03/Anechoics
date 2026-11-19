# -*- coding: utf-8 -*-
"""
Coefficient de Robin alpha(omega) pour un absorbant HYBRIDE ferrite + mousse
pyramidale chargée en carbone, posé sur une paroi métallique.

CONVENTIONS (identiques au code de l'EI, vérifiées dans processing.py)
---------------------------------------------------------------------
* Équation dans l'air : Delta u + k^2 u = 0.
* Condition de Robin telle qu'assemblée par compute_robin_condition :
        du/dn = alpha * u        (n = normale sortante de l'air, vers le mur)
  La condition de Sommerfeld du code est alpha = -i k : l'onde sortante est
  donc exp(-i k x), soit la convention temporelle exp(+j omega t).
* Matériaux à pertes : eps = eps' - j eps'', mu = mu' - j mu'' (eps'', mu'' >= 0),
  comme dans les slides du cours.

MODÈLE (incidence normale, empilement de couches planes)
--------------------------------------------------------
Dans chaque couche : k_m = k0 * sqrt(eps_r * mu_r).
La quantité gamma = (1/mu_r) (du/dx) / u est continue aux interfaces
(continuité de E_tangentiel et de H_tangentiel).
* Sur le métal : u = 0.
* Couche d'épaisseur d posée sur le métal : gamma = -(k_m/mu_r) cot(k_m d).
* Couche suivante : formule de transfert (voir _transfer).
Côté air (mu = 1) : alpha = gamma à la surface.

Cas de vérification :
* matériau semi-infini : alpha -> -j k_m / mu_r ;
* air seul (eps = mu = 1) : alpha = -j k (Sommerfeld), R = 0 ;
* alpha = 0 : R = 1 (réflexion totale).

Mousse pyramidale : approximation de milieu effectif (couches minces).
À la hauteur z (mesurée depuis la pointe), la fraction de mousse dans une
section vaut (z/h)^2 pour une pyramide à base carrée, et
eps_eff = 1 + fraction * (eps_mousse - 1).
C'est une approximation de premier ordre, valable quand la période des
pyramides est petite devant la longueur d'onde.
"""

import numpy

C0 = 299792458.0
EPS0 = 8.8541878128e-12


# ----------------------------------------------------------------------------
# Lois de matériaux
# ----------------------------------------------------------------------------
def mu_ferrite(f, K=2000.0, fm=3.0e6):
    """Perméabilité d'une ferrite NiZn, modèle de relaxation (Debye) :
    mu_r = 1 + K / (1 + j f / fm), convention exp(+j omega t) => mu'' >= 0.
    K  : susceptibilité statique (perméabilité initiale - 1),
    fm : fréquence de relaxation [Hz]."""
    return 1.0 + K / (1.0 + 1j * f / fm)


def eps_lossy(f, eps_p, sigma):
    """Permittivité d'un diélectrique conducteur : eps' - j sigma/(omega eps0)."""
    return eps_p - 1j * sigma / (2.0 * numpy.pi * f * EPS0)


# ----------------------------------------------------------------------------
# Empilement
# ----------------------------------------------------------------------------
def _transfer(gamma_in, k, mu, d):
    """gamma à l'entrée d'une couche (côté air), connaissant gamma côté métal."""
    if gamma_in is None:  # couche posée directement sur le métal
        return -(k / mu) / numpy.tan(k * d)
    t = mu * gamma_in / k
    s, c = numpy.sin(k * d), numpy.cos(k * d)
    return (k / mu) * (s + t * c) / (c - t * s)


def alpha_stack(f, layers):
    """alpha (en m^-1, convention du code) pour un empilement.

    layers : liste de (epaisseur [m], eps_r, mu_r), DU MÉTAL VERS L'AIR.
             eps_r et mu_r peuvent être des fonctions de f.
    """
    k0 = 2.0 * numpy.pi * f / C0
    gamma = None
    for d, eps, mu in layers:
        e = eps(f) if callable(eps) else eps
        m = mu(f) if callable(mu) else mu
        k = k0 * numpy.sqrt(e * m)
        gamma = _transfer(gamma, k, m, d)
    return gamma


def pyramid_layers(height, eps_foam, n_sub=60):
    """Mousse pyramidale (base comprise dans `height`) discrétisée en n_sub
    couches, du métal (base) vers l'air (pointe)."""
    dz = height / n_sub
    layers = []
    for i in range(n_sub):
        z = height - (i + 0.5) * dz          # distance à la pointe
        frac = (z / height) ** 2
        layers.append((dz, (lambda f, fr=frac: 1.0 + fr * (eps_foam(f) - 1.0)), 1.0))
    return layers


def hybrid_layers(d_ferrite=5.5e-3, K=2000.0, fm=3.0e6, eps_ferrite=12.0,
                  h_foam=0.30, foam_eps_p=1.5, foam_sigma=0.05, d_gap=0.0):
    """Empilement hybride : métal | ferrite | (lame d'air) | mousse pyramidale."""
    layers = [(d_ferrite, eps_ferrite, (lambda f: mu_ferrite(f, K, fm)))]
    if d_gap > 0:
        layers.append((d_gap, 1.0, 1.0))
    if h_foam > 0:
        layers += pyramid_layers(h_foam, lambda f: eps_lossy(f, foam_eps_p, foam_sigma))
    return layers


# ----------------------------------------------------------------------------
# Réflexion
# ----------------------------------------------------------------------------
def reflection(alpha, f, theta=0.0):
    """Coefficient de réflexion d'une onde plane, incidence theta (rad),
    pour la condition du/dn = alpha u : R = (alpha + j k cos) / (j k cos - alpha).
    (alpha supposé indépendant de l'angle : modèle de paroi localement réagissante.)"""
    kn = 2.0 * numpy.pi * f / C0 * numpy.cos(theta)
    return (alpha + 1j * kn) / (1j * kn - alpha)


def R_dB(alpha, f, theta=0.0):
    """Réflexion en puissance, en dB (convention des slides : 10 log10 |R|^2)."""
    return 20.0 * numpy.log10(numpy.abs(reflection(alpha, f, theta)))


# ----------------------------------------------------------------------------
# Passage au code de l'EI (grandeurs adimensionnées)
# ----------------------------------------------------------------------------
def to_code_units(f, alpha, L_ref):
    """Le domaine du code a une largeur de 1 unité. Si 1 unité = L_ref mètres :
    wavenumber_code = k0 * L_ref  et  Alpha_code = alpha * L_ref."""
    k0 = 2.0 * numpy.pi * f / C0
    return k0 * L_ref, alpha * L_ref


def compute_alpha_mousse(wavenumber, L_ref=1.0, h_foam=0.30, foam_eps_p=1.5, foam_sigma=0.003):
    """Alpha (unités du code) pour la mousse pyramidale SEULE posée sur le métal
    (pas de ferrite). wavenumber est le nombre d'onde du code (1 unité = L_ref m)."""
    f = wavenumber * C0 / (2.0 * numpy.pi * L_ref)
    a = alpha_stack(f, pyramid_layers(h_foam, lambda ff: eps_lossy(ff, foam_eps_p, foam_sigma)))
    return a * L_ref


def compute_alpha_ls26(wavenumber, L_ref=1.0, eps_p=9.6, tan_delta=1.14):
    """Alpha (unités du code) pour le LS26 (Eccosorb LS-26) SEMI-INFINI, incidence normale.
    Fiche : eps_r = 9.6, tan(delta) = 1.14 supposée constante en fréquence,
    soit sigma = 2 pi f eps0 eps_r tan(delta) = 6.088e-10 * f  [S/m].
    eps = eps'(1 - j tan delta) (convention exp(+j omega t)), mu_r = 1.
    alpha = -j k_m avec k_m = k0 sqrt(eps)  (cf. cas de vérification en tête de fichier)."""
    eps = eps_p * (1.0 - 1j * tan_delta)
    k0 = wavenumber / L_ref                      # nombre d'onde dans l'air [m^-1]
    return -1j * k0 * numpy.sqrt(eps) * L_ref


def compute_alpha(f, L_ref=1.0, **hybrid_params):
    """Interface prête pour demo_control_polycopie2024.py :
        wavenumber, Alpha = compute_alpha.compute_alpha(f, L_ref=3.0)
    """
    a = alpha_stack(f, hybrid_layers(**hybrid_params))
    return to_code_units(f, a, L_ref)
