// ================================================================
// MMSEC — Cours 2 : Notations et Concepts Fondamentaux
// Restructuré pour Typsite | Langue : Français
// ================================================================

#set document(
  title: "MMSEC — Notations et Concepts Fondamentaux",
  author: "S. Voloshynovskiy (cours) / restructuré",
)

#set page(
  paper: "a4",
  margin: (top: 2.5cm, bottom: 2.5cm, left: 3cm, right: 3cm),
)

#set text(lang: "fr", size: 11pt)
#set heading(numbering: none)
#set par(justify: true, leading: 0.8em, spacing: 1.2em)

// ---- Imports ----
#import "@preview/fletcher:0.5.1" as fletcher: diagram, node, edge
#import "@preview/cetz:0.2.2": canvas, draw

// ================================================================
// BOÎTES PÉDAGOGIQUES
// ================================================================

#let def-box(titre, corps) = block(
  width: 100%,
  inset: 12pt,
  radius: 5pt,
  fill: rgb("#dbeafe"),
  stroke: rgb("#3b82f6"),
)[
  *📘 Définition — #titre* \
  #v(4pt)
  #corps
]

#let remarque(corps) = block(
  width: 100%,
  inset: 12pt,
  radius: 5pt,
  fill: rgb("#fef3c7"),
  stroke: rgb("#f59e0b"),
)[
  *💡 Remarque :* #corps
]

#let attention-box(corps) = block(
  width: 100%,
  inset: 12pt,
  radius: 5pt,
  fill: rgb("#fee2e2"),
  stroke: rgb("#ef4444"),
)[
  *⚠️ Important :* #corps
]

#let insight-box(corps) = block(
  width: 100%,
  inset: 12pt,
  radius: 5pt,
  fill: rgb("#f3e8ff"),
  stroke: rgb("#a855f7"),
)[
  *🔑 Insight clé :* #corps
]

#let exemple-box(corps) = block(
  width: 100%,
  inset: 12pt,
  radius: 5pt,
  fill: rgb("#dcfce7"),
  stroke: rgb("#22c55e"),
)[
  *✏️ Exemple :* #corps
]

// ================================================================
// PAGE DE TITRE
// ================================================================

#align(center)[
  #v(3cm)

  #rect(
    width: 90%,
    inset: 30pt,
    radius: 8pt,
    fill: rgb("#1e3a5f"),
    stroke: none,
  )[
    #text(size: 26pt, weight: "bold", fill: white)[
      Notations et Concepts Fondamentaux
    ]
    #v(8pt)
    #text(size: 14pt, fill: rgb("#93c5fd"))[
      _Multimedia Security and Privacy_ — Cours 2
    ]
  ]

  #v(1.5cm)

  #text(size: 12pt)[
    Ce cours établit les *bases mathématiques communes* à toutes les thématiques \ du cours MMSEC. Une référence à consulter tout au long du semestre.
  ]

  #v(1cm)

  #grid(
    columns: (1fr, 1fr, 1fr, 1fr),
    column-gutter: 8pt,
    rect(inset: 10pt, radius: 4pt, fill: rgb("#eff6ff"), stroke: rgb("#93c5fd"))[
      #align(center)[
        *Tatouage \ Numérique*
      ]
    ],
    rect(inset: 10pt, radius: 4pt, fill: rgb("#f0fdf4"), stroke: rgb("#86efac"))[
      #align(center)[
        *Hachage \ Perceptuel*
      ]
    ],
    rect(inset: 10pt, radius: 4pt, fill: rgb("#fdf4ff"), stroke: rgb("#d8b4fe"))[
      #align(center)[
        *Robustesse \ Adversariale*
      ]
    ],
    rect(inset: 10pt, radius: 4pt, fill: rgb("#fff7ed"), stroke: rgb("#fdba74"))[
      #align(center)[
        *Protection \ Vie Privée*
      ]
    ],
  )

  #v(3cm)
]

#pagebreak()

// ================================================================
// SOMMAIRE
// ================================================================

#outline(
  title: "Sommaire",
  depth: 2,
  indent: 1.5em,
)

#pagebreak()

// ================================================================
= Introduction et Motivation
// ================================================================

== Pourquoi ce cours fondamental ?

La *sécurité multimédia* (_Multimedia Security_) repose sur des concepts mathématiques
partagés entre plusieurs domaines. Ce cours introduit un *vocabulaire commun* pour
aborder avec rigueur les thématiques suivantes :

#grid(
  columns: (auto, 1fr),
  column-gutter: 1em,
  row-gutter: 0.8em,
  rect(fill: rgb("#dbeafe"), inset: 6pt, radius: 3pt)[*Tatouage numérique*],
  [Intégration et détection de marques invisibles dans du contenu],
  rect(fill: rgb("#dcfce7"), inset: 6pt, radius: 3pt)[*Hachage perceptuel*],
  [Génération de représentations compactes et robustes],
  rect(fill: rgb("#fdf4ff"), inset: 6pt, radius: 3pt)[*Robustesse adversariale*],
  [Analyse et défense contre les attaques adversariales],
  rect(fill: rgb("#fff7ed"), inset: 6pt, radius: 3pt)[*Protection de la vie privée*],
  [Conformité aux réglementations, anonymisation],
)

== Plan du cours

Ce cours est organisé en *quatre parties progressives* :

+ *Notations de base et représentation des images* — Comment représenter mathématiquement une image
+ *Fondements du Deep Learning* — Apprentissage auto-supervisé pour l'extraction de _features_
+ *Notations pour le tatouage et le hachage perceptuel* — Formalisation des problèmes clés
+ *Détection de signal dans le bruit* — Fondements probabilistes de la détection

// ================================================================
= Partie 1 : Représentation des Images
// ================================================================

== Notations fondamentales

=== Variables et symboles principaux

Tout au long de ce cours, nous utilisons les notations suivantes de façon cohérente :

#grid(
  columns: (0.2fr, 0.8fr),
  column-gutter: 1.5em,
  row-gutter: 0.8em,
  align(center)[$bold(x)$], [*Image originale* (avant tatouage)],
  align(center)[$bold(y)$], [*Image _stégo_* : image contenant le tatouage],
  align(center)[$bold(w)$], [*Tatouage* (_watermark_) : le signal à intégrer],
  align(center)[$bold(k)$], [*Clé secrète* utilisée pour l'intégration ou le décodage],
  align(center)[$bold(m)$], [*Message multi-bit* : $bold(m) in {0, 1}^L$ où $L$ est la longueur],
)

=== Dimensions d'une image

Une image numérique est un *tenseur tridimensionnel* :

$ H times W times C $

#grid(
  columns: (0.15fr, 0.85fr),
  column-gutter: 1em,
  row-gutter: 0.6em,
  [$H$], [Hauteur de l'image en pixels (_Height_)],
  [$W$], [Largeur de l'image en pixels (_Width_)],
  [$C$], [Nombre de canaux : $C = 1$ (niveaux de gris) ou $C = 3$ (RGB)],
)

=== Espace des valeurs de pixels

Sur *8 bits par canal*, chaque pixel $x_i$ appartient à l'ensemble :

$ cal(X) = {0, 1, dots, 255} $

Car $2^8 = 256$ niveaux distincts sont possibles.

// ================================================================
== Images en niveaux de gris et images couleur
// ================================================================

=== Image en niveaux de gris

#align(center)[
  #canvas({
    import draw: *

    // Image rectangle (gris)
    rect((0, 0), (rel: (4, 2.8)),
      fill: rgb("#e5e7eb"),
      stroke: (paint: rgb("#374151"), thickness: 1.5pt)
    )
    content((2, 1.4), text(size: 10pt, fill: rgb("#374151"))[Image $H times W$])

    // Annotations H et W
    line((-0.5, 0), (-0.5, 2.8),
      mark: (start: ">", end: ">"),
      stroke: rgb("#3b82f6") + 1.2pt
    )
    content((-1.1, 1.4), text(fill: rgb("#3b82f6"))[$H$])

    line((0, -0.4), (4, -0.4),
      mark: (start: ">", end: ">"),
      stroke: rgb("#3b82f6") + 1.2pt
    )
    content((2, -0.8), text(fill: rgb("#3b82f6"))[$W$])

    // Flèche vers vecteur
    line((4.4, 1.4), (5.4, 1.4), mark: (end: ">"), stroke: 1.5pt)
    content((4.9, 1.7), text(size: 9pt)[aplatir])

    // Vecteur colonne
    rect((5.6, 0.2), (6.2, 2.6),
      fill: rgb("#dbeafe"),
      stroke: rgb("#3b82f6") + 1.2pt
    )
    content((5.9, 1.4), text(fill: rgb("#1d4ed8"))[$bold(x)$])

    // Annotation N
    line((6.4, 0.2), (6.4, 2.6),
      mark: (start: ">", end: ">"),
      stroke: rgb("#9ca3af") + 1pt
    )
    content((7.2, 1.4), text(size: 9pt)[$N = H times W$])
  })
]

- Chaque pixel : $x_i in {0, dots, 255}$
- Taille totale de l'image : $(H times W) times 256$ combinaisons possibles

=== Image RGB (couleur)

Une image couleur est composée de *trois canaux superposés* : Rouge (R), Vert (G), Bleu (B).

#align(center)[
  #canvas({
    import draw: *

    // Canal B (derrière)
    rect((0.8, 0.8), (4.8, 3.6),
      fill: rgb("#bfdbfe"),
      stroke: (paint: rgb("#1d4ed8"), thickness: 1.5pt)
    )
    content((4.6, 3.3), text(fill: rgb("#1d4ed8"))[B])

    // Canal G (milieu)
    rect((0.4, 0.4), (4.4, 3.2),
      fill: rgb("#bbf7d0"),
      stroke: (paint: rgb("#16a34a"), thickness: 1.5pt)
    )
    content((4.2, 2.9), text(fill: rgb("#16a34a"))[G])

    // Canal R (devant)
    rect((0, 0), (4, 2.8),
      fill: rgb("#fecaca"),
      stroke: (paint: rgb("#dc2626"), thickness: 1.5pt)
    )
    content((3.8, 2.5), text(fill: rgb("#dc2626"))[R])

    // Annotations
    line((-0.5, 0), (-0.5, 2.8),
      mark: (start: ">", end: ">"),
      stroke: rgb("#374151") + 1pt
    )
    content((-1.0, 1.4), [$H$])

    line((0, -0.4), (4, -0.4),
      mark: (start: ">", end: ">"),
      stroke: rgb("#374151") + 1pt
    )
    content((2, -0.8), [$W$])

    // Formule
    content((6.2, 2.0), $x_i^(R,G,B) in {0, dots, 255}$)
    content((6.2, 1.2), text(size: 9pt)[Taille : $(H times W) times 3 times 256$])
  })
]

=== Représentation vectorisée

Pour les opérations mathématiques, on *aplatit* (_flatten_) l'image en vecteur :

- *Image niveaux de gris* :
  $ bold(x) in RR^(H dot W), quad "avec" M = H dot W "pixels au total" $

- *Image RGB* :
  $ bold(x)_"RGB" in RR^(3 dot H dot W) $

#remarque[
  En Python, la notation vectorielle n'est pas explicite. Les dimensions sont précisées par le contexte ou via `tensor.shape`. Par exemple, une image $256 times 256 times 3$ peut être aplatie en un vecteur de dimension $196608$.
]

// ================================================================
== Sources d'images modernes
// ================================================================

=== Résolutions typiques

Les appareils photo modernes produisent des images à très haute résolution :

#grid(
  columns: (1fr, 1fr),
  column-gutter: 1em,
  rect(inset: 12pt, radius: 4pt, fill: rgb("#f0fdf4"), stroke: rgb("#86efac"))[
    *Smartphones (ex. iPhone)*
    - Résolution : 12–24 mégapixels
    - Format courant : $4032 times 3024 times 3$
  ],
  rect(inset: 12pt, radius: 4pt, fill: rgb("#eff6ff"), stroke: rgb("#93c5fd"))[
    *Hauts de gamme (ex. Samsung)*
    - Résolution : jusqu'à 108 mégapixels
    - Format : $12000 times 9000 times 3$
  ],
)

=== Pipeline de traitement : du capteur à l'image RGB

Les capteurs d'images utilisent un *motif de Bayer* (_Bayer Pattern_) : chaque photosite ne capte qu'une seule couleur. Le *_demosaicking_* reconstitue les trois canaux.

#align(center)[
  #diagram(
    node-stroke: 1.2pt,
    node-fill: white,
    spacing: (5em, 0em),
    node((0, 0), [Capteur \ (motif Bayer)], shape: rect, fill: rgb("#fef3c7")),
    edge("->", label: [_Demosaicking_], label-side: center),
    node((1, 0), [Image RGB \ $H times W times 3$], shape: rect, fill: rgb("#dcfce7")),
  )
]

- *Motif de Bayer* : grille alternant pixels R, G, B (2× plus de pixels verts pour mimer l'œil humain)
- *_Demosaicking_* : interpolation des valeurs manquantes à chaque position

#attention-box[
  Le *format d'entrée standard* pour ce cours est toujours une image RGB de taille $H times W times 3$.
]

// ================================================================
== Métriques de qualité perceptuelle
// ================================================================

Ces métriques quantifient la *distorsion* introduite entre une image originale $bold(x)$ et une image modifiée $bold(y)$ (après tatouage par exemple).

=== Erreur Quadratique Moyenne (MSE)

#def-box("MSE — _Mean Squared Error_")[
  $ "MSE"(bold(x), bold(y)) = 1/M ||bold(x) - bold(y)||_2^2 = 1/M sum_(i=1)^M (x_i - y_i)^2 $

  Mesure la *distorsion moyenne au carré* entre les pixels correspondants.
]

=== Rapport Signal sur Bruit de Crête (PSNR)

#def-box("PSNR — _Peak Signal-to-Noise Ratio_")[
  $ "PSNR" = 10 dot log_10 lr((frac("MAX"^2, "MSE"(bold(x), bold(y))))) $

  - $"MAX" = 255$ pour des images 8 bits
  - Plus le PSNR est *élevé*, meilleure est la qualité visuelle
]

#exemple-box[
  - PSNR > 40 dB → distorsion imperceptible
  - PSNR ~ 30 dB → légère dégradation visible
  - PSNR < 20 dB → forte dégradation
]

=== Versions pondérées (WMSE / WPSNR)

Ces versions tiennent compte de la *sensibilité perceptuelle* de l'œil humain :

$ "WMSE"(bold(x), bold(y)) = 1/M sum_(i=1)^M w_i dot (x_i - y_i)^2 $

$ "WPSNR" = 10 dot log_10 lr((frac("MAX"^2, "WMSE"(bold(x), bold(y))))) $

où $w_i$ est une *fonction de pondération* (par exemple, donner plus de poids aux zones texturées où les modifications sont moins visibles).

// ================================================================
== Métriques de similarité vectorielle
// ================================================================

Ces métriques comparent deux vecteurs (représentations d'images, _embeddings_, etc.).

=== Produit scalaire

$ bold(x)^top bold(y) = sum_(i=1)^M x_i y_i $

Mesure la *corrélation linéaire* entre deux vecteurs.

=== Similarité cosinus

#def-box("Similarité Cosinus")[
  $ cos(theta) = frac(bold(x)^top bold(y), ||bold(x)||_2 dot ||bold(y)||_2) in [-1, 1] $

  Mesure l'*angle* entre deux vecteurs, indépendamment de leur norme.
]

#grid(
  columns: (1fr, 1fr, 1fr),
  column-gutter: 1em,
  rect(inset: 10pt, radius: 4pt, fill: rgb("#dcfce7"), stroke: rgb("#22c55e"), align: center)[
    $cos theta = 1$ \
    Même direction \ (vecteurs identiques)
  ],
  rect(inset: 10pt, radius: 4pt, fill: rgb("#fef3c7"), stroke: rgb("#f59e0b"), align: center)[
    $cos theta = 0$ \
    Orthogonaux \ (non corrélés)
  ],
  rect(inset: 10pt, radius: 4pt, fill: rgb("#fee2e2"), stroke: rgb("#ef4444"), align: center)[
    $cos theta = -1$ \
    Directions opposées
  ],
)

#remarque[
  La *distance euclidienne* $||bold(x) - bold(y)||_2$ et la similarité cosinus sont complémentaires : la première mesure l'écart absolu, la seconde mesure l'angle.
]

// ================================================================
= Partie 1.3 : Représentation dans le Domaine Transformé
// ================================================================

== Vue d'ensemble des transformées d'images

Une *transformée* $cal(T)$ convertit une image de l'espace pixel vers un autre espace (fréquentiel, etc.) :

$ bold(z) = cal(T)(bold(x)), quad bold(x) = cal(T)^(-1)(bold(z)) $

#align(center)[
  #diagram(
    node-stroke: 1pt,
    node-fill: white,
    spacing: (5em, 0em),
    node((0, 0), [$bold(x)$ \ Espace pixel], shape: rect, fill: rgb("#dbeafe")),
    edge("->", label: [$cal(T)$], label-side: center),
    node((1, 0), [$bold(z)$ \ Espace transformé], shape: rect, fill: rgb("#fdf4ff")),
    edge("->", label: [$cal(T)^(-1)$], label-side: center),
    node((2, 0), [$bold(x)$ \ Espace pixel], shape: rect, fill: rgb("#dbeafe")),
  )
]

Le cours couvre *quatre grandes catégories* de transformées :

#grid(
  columns: (0.15fr, 0.4fr, 0.45fr),
  column-gutter: 0.8em,
  row-gutter: 0.8em,
  [*N°*], [*Méthode*], [*Caractéristiques*],
  [*1*], [Décomposition _bit-plane_], [Représentation binaire des pixels],
  [*2*], [Transformées orthogonales (DFT, DCT, DWT)], [Fixes, analytiques, inversibles],
  [*3*], [ACP (_PCA_)], [Apprenante linéaire, données-dépendante],
  [*4*], [DL / _Foundation Models_], [Apprenante non-linéaire, très puissante],
)

// ================================================================
== Décomposition en plans de bits (_Bit-Plane_)
// ================================================================

=== Principe

Chaque pixel sur *8 bits* se décompose en une somme de puissances de 2 :

$ x = x_0 dot 2^0 + x_1 dot 2^1 + x_2 dot 2^2 + dots + x_7 dot 2^7 $

avec $x_i in {0, 1}$ le $i$-ème bit du pixel.

Cette décomposition produit *8 images binaires*, une par plan de bits.

=== Importance des plans

#grid(
  columns: (1fr, 1fr),
  column-gutter: 1em,
  rect(inset: 10pt, radius: 4pt, fill: rgb("#fee2e2"), stroke: rgb("#ef4444"))[
    *MSB — Bit de poids fort (plan 7)*
    - Porte $2^7 = 128$ de valeur
    - Contient l'*essentiel* de l'information visuelle
    - L'image originale y est reconnaissable
  ],
  rect(inset: 10pt, radius: 4pt, fill: rgb("#dcfce7"), stroke: rgb("#22c55e"))[
    *LSB — Bit de poids faible (plan 0)*
    - Porte seulement $2^0 = 1$ de valeur
    - Apparaît comme du *bruit aléatoire*
    - Modifications *imperceptibles*
  ],
)

=== Applications

- *Compression* : on peut ignorer les LSB sans perte visuelle notable
- *Tatouage LSB* : intégrer un message dans les LSB (invisible mais fragile)
- *Analyse forensique* : détecter des manipulations dans les plans de bits

// ================================================================
== Transformée de Fourier Discrète (DFT)
// ================================================================

=== Définition mathématique

#def-box("DFT — _Discrete Fourier Transform_")[
  *Transformée directe* (espace pixel → fréquentiel) :
  $ z[k] = sum_(n=0)^(N-1) x[n] dot e^(-j 2 pi k n \/ N), quad k = 0, dots, N-1 $

  *Transformée inverse* (fréquentiel → espace pixel) :
  $ x[n] = 1/N sum_(k=0)^(N-1) z[k] dot e^(j 2 pi k n \/ N) $
]

=== Propriétés fondamentales

- *Linéarité* : $cal(T)(alpha bold(x) + beta bold(y)) = alpha cal(T)(bold(x)) + beta cal(T)(bold(y))$
- *Conservation de l'énergie* (Parseval) : $sum_n |x[n]|^2 = 1/N sum_k |z[k]|^2$
- *Périodicité* : la DFT est périodique de période $N$
- *Invariance en magnitude* : $|z[k]|$ est invariant par translation de $bold(x)$

=== Structure de la sortie

Les coefficients DFT sont *complexes* et se décomposent en :

$ z[k] = underbrace(|z[k]|, "Spectre de magnitude") dot e^(j phi_k) quad "avec" quad phi_k = underbrace(arg(z[k]), "Spectre de phase") $

=== Applications en sécurité multimédia

- *Filtrage fréquentiel* : atténuer ou amplifier certaines fréquences
- *Tatouage en domaine fréquentiel* : intégration plus robuste qu'en LSB
- *Analyse et débruitage* de signaux

// ================================================================
== Transformée en Cosinus Discrète (DCT)
// ================================================================

=== Définition

#def-box("DCT — _Discrete Cosine Transform_")[
  *Transformée directe :*
  $ z[k] = alpha_k sum_(n=0)^(N-1) x[n] dot cos lr((pi/N (n + 0.5) k)) $

  *Transformée inverse :*
  $ x[n] = sum_(k=0)^(N-1) alpha_k z[k] dot cos lr((pi/N (n + 0.5) k)) $

  avec le facteur de normalisation :
  $ alpha_k = cases(sqrt(1/N) quad "si" k = 0, sqrt(2/N) quad "sinon") $
]

=== Trois façons d'appliquer la DCT à une image

#grid(
  columns: (1fr, 1fr, 1fr),
  column-gutter: 8pt,
  rect(inset: 10pt, radius: 4pt, fill: rgb("#eff6ff"), stroke: rgb("#93c5fd"))[
    *1. DCT globale*

    Appliquée à *toute l'image* d'un coup.

    $ bold(z) = "DCT"(bold(x)) $
    $ bold(x) in RR^(H times W) $

    Fournit une vue *globale* des fréquences.
  ],
  rect(inset: 10pt, radius: 4pt, fill: rgb("#f0fdf4"), stroke: rgb("#86efac"))[
    *2. DCT par blocs 8×8*

    Image découpée en blocs. DCT appliquée *indépendamment* à chaque bloc.

    $ bold(z)_(i,j) = "DCT"(bold(x)_(i,j)) $
    $ bold(x)_(i,j) in RR^(8 times 8) $

    Utilisé en *JPEG*.
  ],
  rect(inset: 10pt, radius: 4pt, fill: rgb("#fdf4ff"), stroke: rgb("#d8b4fe"))[
    *3. DCT tensorielle*

    Sous-échantillonnage tous les 8 pixels. DCT sur chaque sous-image.

    $ bold(z)_(i,j) = "DCT"(bold(x)^((i,j))) $
    $ bold(x)^((i,j)) in RR^(H\/8 times W\/8) $
  ],
)

=== Applications

- *Compression JPEG et MPEG* : blocs 8×8, quantification des coefficients
- *Tatouage dans les coefficients DC/AC* : robust car dans le domaine fréquentiel
- *Forensique multimédia* : détection de réencodages JPEG

// ================================================================
== Transformée en Ondelettes Discrètes (DWT)
// ================================================================

=== Principe

La *DWT* (_Discrete Wavelet Transform_) décompose un signal en appliquant récursivement deux filtres :

#align(center)[
  #diagram(
    node-stroke: 1pt,
    node-fill: white,
    spacing: (4em, 2em),
    node((0, 0), [$bold(x)$], shape: circle, fill: rgb("#dbeafe")),
    edge((0,0), (1, 0.8), "->", label: [Passe-bas ($L$)], label-side: left),
    edge((0,0), (1,-0.8), "->", label: [Passe-haut ($H$)], label-side: right),
    node((1, 0.8), [Approximation \ (basses fréq.)], shape: rect, fill: rgb("#dcfce7")),
    node((1,-0.8), [Détails \ (hautes fréq.)], shape: rect, fill: rgb("#fef3c7")),
  )
]

Pour une image 2D, on obtient *4 sous-bandes* : LL (approximation), LH, HL, HH (détails).

=== Propriétés clés

- *Analyse multi-résolution* : capture des détails à différentes *échelles*
- *Localisation spatio-fréquentielle* : contrairement à la DFT, localise *où* est la fréquence
- *Conservation de l'énergie* : reconstruction parfaite par DWT inverse
- *Support compact* : les ondelettes sont localisées en espace

=== Applications

- *JPEG 2000* : compression basée sur DWT (meilleure qualité que JPEG classique)
- *Tatouage multi-résolution* : intégration à différentes échelles
- *Débruitage* : seuillage des coefficients de détail

// ================================================================
== Transformée Apprenante : ACP (PCA)
// ================================================================

=== Intuition

La *PCA* (_Principal Component Analysis_) — ou *Analyse en Composantes Principales* — apprend une *base orthogonale* directement à partir des données, en cherchant les directions de *variance maximale*.

Contrairement à la DFT/DCT qui sont fixes, la PCA *s'adapte* aux données.

=== Algorithme étape par étape

+ *Collecter* $N$ échantillons $bold(x)_1, bold(x)_2, dots, bold(x)_N$

+ *Calculer la moyenne* :
  $ bar(bold(x)) = 1/N sum_(i=1)^N bold(x)_i $

+ *Centrer les données* :
  $ bold(x)'_i = bold(x)_i - bar(bold(x)) $

+ *Calculer la matrice de covariance* :
  $ bold(K)_(x x) = 1/N sum_(i=1)^N bold(x)'_i bold(x)_i^(prime top) $

+ *Décomposition en valeurs propres* :
  $ bold(K)_(x x) = bold(U) bold(Sigma) bold(U)^top $
  où $bold(U)$ contient les *vecteurs propres* et $bold(Sigma)$ les *valeurs propres* (variances).

+ *Transformation* :
  $ bold(z) = bold(U)^top bold(x) quad "et" quad bold(x) = bold(U) bold(z) $

=== Propriété d'orthonormalité

$ bold(U)^top bold(U) = bold(U) bold(U)^top = bold(I) $

La transformation est *parfaitement inversible*, préservant toute l'information.

=== Cas d'usage

- *Décorrélation des données* : les composantes $bold(z)$ sont non corrélées
- *Compression* : conserver les $k$ premières composantes (les plus variantales)
- *Visualisation* : projection en 2D ou 3D des données haute dimension

#attention-box[
  *Limitation fondamentale* : si les données se trouvent sur une *variété non-linéaire* (ex. visages, objets), la PCA échoue. Elle suppose implicitement que les données suivent une distribution *gaussienne multivariée*.

  → Solution : transformées non-linéaires par *deep learning*.
]

// ================================================================
= Partie 2 : Fondements du Deep Learning pour l'Extraction de Features
// ================================================================

== De la PCA aux Transformées Non-Linéaires

=== La limite des méthodes linéaires

#grid(
  columns: (1fr, 1fr),
  column-gutter: 1em,
  rect(inset: 12pt, radius: 5pt, fill: rgb("#fef3c7"), stroke: rgb("#f59e0b"))[
    *Transformées classiques (PCA, DFT, DCT)*
    - Transformations *linéaires* ou *analytiques*
    - *Inversibles* (reconstruction parfaite)
    - *Indépendantes* des données (sauf PCA)
    - Limitées aux structures *linéaires*
  ],
  rect(inset: 12pt, radius: 5pt, fill: rgb("#dbeafe"), stroke: rgb("#3b82f6"))[
    *Transformées DL apprenantes*
    - Transformations *non-linéaires*
    - Pas nécessairement inversibles
    - *Apprises* sur de grandes bases de données
    - Capturent des structures *arbitrairement complexes*
  ],
)

=== L'espace latent idéal

Un bon espace latent (_latent space_) doit être :

- *Représentatif* : captures les features pertinentes du contenu
- *Robuste* : stable face au bruit (compression JPEG, recadrage, etc.)
- *Invariant* : stable face aux transformations géométriques (rotation, mise à l'échelle)

Ces propriétés sont *essentielles* pour le tatouage robuste et le hachage perceptuel.

// ================================================================
== Architecture d'un Encodeur Neuronal
// ================================================================

=== Définition formelle

#def-box("Encodeur neuronal profond")[
  $ f_phi(bold(x)) = sigma_K (bold(W)_K dots (sigma_1 (bold(W)_1 bold(x) + bold(b)_1) + bold(b)_K)) $

  avec les paramètres apprenables : $phi = {bold(W)_1, dots, bold(W)_K ; bold(b)_1, dots, bold(b)_K}$
]

=== Composants d'une couche

#grid(
  columns: (0.2fr, 0.8fr),
  column-gutter: 1em,
  row-gutter: 0.7em,
  [$bold(W)_i$], [*Matrices de poids* — transformations linéaires apprenables],
  [$sigma_i$], [*Fonctions d'activation* non-linéaires (ReLU, sigmoid, GELU...)],
  [$bold(b)_i$], [*Biais* — décalage des activations],
  [$phi$], [*Paramètres* appris par rétropropagation (_backpropagation_)],
)

#insight-box[
  Sans les fonctions d'activation $sigma_i$, empiler plusieurs couches serait équivalent à une *seule couche linéaire* — exactement comme la PCA ! La non-linéarité est ce qui rend les réseaux profonds *universellement puissants*.
]

=== Couches convolutionnelles (CNNs)

Pour les images, les *convolutions* remplacent avantageusement les multiplications matricielles globales :

- Une *fenêtre glissante* (filtre $bold(W)_"filter"$) parcourt l'image
- À chaque position : somme pondérée localement (_produit scalaire_ filtre × patch)
- Résultat : une *carte de features* (_feature map_)

*Avantages* des convolutions :
- *Partage de paramètres* : le même filtre s'applique partout (invariance par translation)
- *Localité* : capture des patterns locaux (bords, textures)
- *Efficacité* : beaucoup moins de paramètres qu'une couche dense

=== Image _Patching_ pour les Transformeurs (ViT)

Les *Vision Transformers* (_ViT_) découpent l'image en *patches* non-chevauchants :

#align(center)[
  #diagram(
    node-stroke: 1pt,
    node-fill: white,
    spacing: (5em, 0em),
    node((0, 0), [Image \ $H times W$], shape: rect, fill: rgb("#dbeafe")),
    edge("->", label: [Découpage], label-side: center),
    node((1, 0), [Patches \ $P_1, P_2, dots$], shape: rect, fill: rgb("#fef3c7")),
    edge("->", label: [Projection \ linéaire], label-side: center),
    node((2, 0), [Tokens \ $T_1, T_2, dots$], shape: rect, fill: rgb("#dcfce7")),
    edge("->", label: [Transformeur], label-side: center),
    node((3, 0), [Représentation \ latente $tilde(bold(y))$], shape: rect, fill: rgb("#fdf4ff")),
  )
]

// ================================================================
== Apprentissage Auto-Supervisé (SSL)
// ================================================================

=== Principe général

L'*apprentissage auto-supervisé* (_Self-Supervised Learning_, SSL) entraîne un encodeur *sans annotations humaines*, en utilisant les *augmentations* de données comme signal d'apprentissage.

*Idée centrale* : deux vues différentes de la même image doivent avoir des représentations *proches* dans l'espace latent.

=== Processus d'entraînement

#align(center)[
  #diagram(
    node-stroke: 1pt,
    node-fill: white,
    spacing: (4em, 2.5em),
    node((0, 0), [$bold(x)_0$ \ Image originale], shape: rect, fill: rgb("#e0f2fe")),
    edge((0,0), (1, 0.8), "->", label: [augm. 1], label-side: left),
    edge((0,0), (1,-0.8), "->", label: [augm. 2], label-side: right),
    node((1, 0.8), [$bold(x)_+^((1))$], shape: circle, fill: rgb("#fef3c7")),
    node((1,-0.8), [$bold(x)_+^((2))$], shape: circle, fill: rgb("#fef3c7")),
    edge((1,0.8), (2, 0.8), "->", label: [$f_phi$], label-side: center),
    edge((1,-0.8), (2,-0.8), "->", label: [$f_phi$], label-side: center),
    node((2, 0.8), [$tilde(bold(y))_1$], shape: circle, fill: rgb("#bbf7d0")),
    node((2,-0.8), [$tilde(bold(y))_2$], shape: circle, fill: rgb("#bbf7d0")),
    edge((2,0.8), (3, 0), "->"),
    edge((2,-0.8), (3, 0), "->"),
    node((3, 0), [Maximiser \ $cal(L)_y(tilde(bold(y))_1, tilde(bold(y))_2)$], shape: rect, fill: rgb("#fdf4ff")),
  )
]

=== Propriétés clés du SSL

- *Pas de données labelisées* : adapté aux grandes bases non annotées
- *Augmentations* : recadrage, rotations, changements de couleur, flou, etc.
- *Espace latent robuste* : invariant aux transformations appliquées

// ================================================================
== Architectures de _Foundation Models_ SSL
// ================================================================

=== Les trois paradigmes d'entraînement

Les _Foundation Models_ modernes utilisent trois grandes familles d'architectures :

==== 1. Reconstruction (Auto-Encodeurs / MAE)

L'encodeur compresse l'image, un décodeur tente de la reconstruire :

#align(center)[
  #diagram(
    node-stroke: 1pt,
    spacing: (4em, 0em),
    node((0,0), [$bold(x)$], shape: rect, fill: rgb("#dbeafe")),
    edge("->", label: [Encodeur $f_psi$]),
    node((1,0), [$tilde(bold(y))$], shape: rect, fill: rgb("#fdf4ff")),
    edge("->", label: [Décodeur $g_theta$]),
    node((2,0), [$hat(bold(x))$], shape: rect, fill: rgb("#dcfce7")),
    edge((2,0), (2,-0.8), "->"),
    node((2,-0.8), [$cal(L)_x(bold(x), hat(bold(x)))$], shape: rect, fill: rgb("#fef3c7")),
  )
]

- ✅ Pas d'effondrement de mode (_mode collapse_)
- ❌ Haute complexité computationnelle
- ❌ Difficulté à définir une bonne _loss_ dans l'espace pixel

_Exemples_ : *MAE* (_Masked Autoencoder_), *Denoising-AE*

==== 2. Embedding Conjoint (_Joint Embedding_)

Deux vues de la même image sont encodées et leurs représentations *maximisent leur similarité* :

$ cal(L)_y(tilde(bold(y))_x, tilde(bold(y))_(x_+)) $

- ✅ Faible complexité
- ✅ Bonnes fonctions de _loss_ dans l'espace latent
- ❌ Risque d'*effondrement de mode* (toutes les représentations s'égalisent)

_Exemples_ : *SimCLR*, *BYOL*, *DINO*, *VICReg*, *BarlowTwins*

==== 3. Prédiction d'Embedding (_Joint Embedding-Prediction_)

Un *prédicteur* $p_eta$ prédit la représentation d'une vue à partir de l'autre :

$ cal(L)_y(tilde(bold(y))_x, hat(bold(y))_(x_+)) $

- ✅ Complexité modérée
- ✅ Évite partiellement l'effondrement de mode

_Exemples_ : *I-JEPA*, *World Model*

=== Apprentissage Faiblement Supervisé (WSL)

Le *WSL* (_Weakly Supervised Learning_) aligne deux modalités (image + texte) :

$ bold(y)_x = f_(phi_x)(bold(x)_0), quad bold(y)_z = f_(phi_z)(bold(z)) $

*Objectif* : maximiser la similarité $cal(L)_y(bold(y)_x, bold(y)_z)$ pour des paires image-texte alignées.

_Exemples_ : *CLIP*, *GLIP*, *BLIP*, *SigLIP*, *CoCa*, *ALIGN*

// ================================================================
= Partie 3 : Notations pour le Tatouage et le Hachage Perceptuel
// ================================================================

== Fonctions d'intégration de tatouage

=== Tatouage zéro-bit

Le *tatouage zéro-bit* (_zero-bit watermarking_) teste uniquement la *présence ou l'absence* d'un tatouage, sans message associé :

#def-box("Tatouage zéro-bit")[
  $ bold(y) = phi(bold(x), bold(k)) $

  - $bold(x)$ : image originale
  - $bold(k)$ : clé secrète (définit la forme du tatouage)
  - $bold(y)$ : image _stégo_ (image tatouée)
]

=== Tatouage multi-bits

Le *tatouage multi-bits* (_multi-bit watermarking_) intègre un *message* de $L$ bits :

#def-box("Tatouage multi-bit")[
  $ bold(y) = phi(bold(x), bold(k), bold(m)) $

  - $bold(m) in {0, 1}^L$ : message de $L$ bits (ex. identifiant de propriétaire)
  - $bold(k)$ : clé secrète
  - $bold(y)$ : image _stégo_
]

=== Décodage

Pour extraire le message de l'image _stégo_ (éventuellement dégradée) :

$ hat(bold(m)) = psi(tilde(bold(y)), bold(k)) $

où $tilde(bold(y))$ est l'image _stégo_ après d'éventuelles attaques (compression, bruit...).

#remarque[
  Les notations $bold(x), bold(y), bold(w), bold(k), bold(m)$ sont utilisées de façon *cohérente* dans tous les cours suivants sur le tatouage, le hachage perceptuel et la robustesse adversariale.
]

// ================================================================
= Partie 4 : Fondements de la Détection de Signal dans le Bruit
// ================================================================

// ================================================================
== Formulation du problème de détection
// ================================================================

=== Contexte

La *détection de tatouage* s'inscrit dans le cadre général du *test d'hypothèses binaire* : étant donnée une image observée $bold(y)$, décider si elle contient un tatouage ou non.

=== Les deux hypothèses

#grid(
  columns: (1fr, 1fr),
  column-gutter: 1em,
  rect(inset: 14pt, radius: 5pt, fill: rgb("#dbeafe"), stroke: rgb("#3b82f6"))[
    *Hypothèse nulle $H_0$*

    $ bold(y) = bold(x) $

    *Aucun tatouage* présent.
    L'image observée est l'image originale.
  ],
  rect(inset: 14pt, radius: 5pt, fill: rgb("#fee2e2"), stroke: rgb("#ef4444"))[
    *Hypothèse alternative $H_1$*

    $ bold(y) = bold(x) + bold(w) $

    *Tatouage présent*.
    L'image originale a été modifiée.
  ],
)

=== Hypothèses de modélisation

Pour dériver les formules analytiques :

- $bold(x)$ : vecteur gaussien *centré*, de covariance $sigma_x^2 bold(I)_M$

  $ bold(x) tilde cal(N)(bold(0), sigma_x^2 bold(I)_M) $

- $bold(w)$ : tatouage *connu*, avec énergie fixée $||bold(w)||_2^2 = E_w$
- $M = H times W$ : nombre total de pixels

=== Objectifs

À partir de ces hypothèses, on veut calculer :

- *$P_("FA")(gamma)$* : probabilité de fausse acceptation (_False Acceptance_)
- *$P_D(gamma)$* : probabilité de détection correcte
- *$P_("miss")(gamma) = 1 - P_D(gamma)$* : probabilité de manquer le tatouage

// ================================================================
== Rapport de vraisemblance et statistique de test
// ================================================================

=== Règle de décision optimale (LRT)

#def-box("Rapport de Vraisemblance (LRT — _Likelihood Ratio Test_)")[
  $ Lambda(bold(y)) = frac(p(bold(y) | H_1), p(bold(y) | H_0)) attach(>=, t: H_1, b: H_0) gamma $

  Comparer les probabilités que l'observation soit issue de $H_1$ vs $H_0$.
]

=== Calcul des densités

*Sous $H_0$* ($bold(y) = bold(x)$, gaussien centré) :

$ p(bold(y)|H_0) = frac(1, (2 pi sigma_x^2)^(M\/2)) exp lr((-frac(bold(y)^top bold(y), 2 sigma_x^2))) $

*Sous $H_1$* ($bold(y) = bold(x) + bold(w)$, gaussien décalé par $bold(w)$) :

$ p(bold(y)|H_1) = frac(1, (2 pi sigma_x^2)^(M\/2)) exp lr((-frac((bold(y) - bold(w))^top (bold(y) - bold(w)), 2 sigma_x^2))) $

=== Simplification du log-rapport

En prenant le *logarithme* du rapport de vraisemblance :

$ ln Lambda(bold(y)) = frac(2 bold(y)^top bold(w) - bold(w)^top bold(w), 2 sigma_x^2) $

Le terme $bold(w)^top bold(w)$ est une *constante* (le tatouage $bold(w)$ est connu). La règle de décision se réduit donc à comparer :

$ bold(y)^top bold(w) attach(>=, t: H_1, b: H_0) gamma' $

=== Statistique de test (score)

#def-box("Statistique de test — Score $S$")[
  $ S(bold(y)) = 1/M bold(y)^top bold(w) = 1/M sum_(i=1)^M y_i w_i $

  Le score est le *produit scalaire normalisé* entre l'image observée et le tatouage de référence.
]

*Interprétation intuitive* : si $bold(y) = bold(x) + bold(w)$, alors $bold(y)^top bold(w)$ sera grand (le tatouage corrèle avec lui-même). Si $bold(y) = bold(x)$ (pas de tatouage), le produit scalaire sera proche de zéro (bruit non corrélé avec $bold(w)$).

// ================================================================
== Distribution du score sous chaque hypothèse
// ================================================================

=== Calcul sous $H_0$

*Moyenne* sous $H_0$ : puisque $bold(y) = bold(x)$ et $EE[bold(x)] = bold(0)$ :

$ mu_0 = EE[S(bold(y)) | H_0] = 1/M bold(w)^top EE[bold(x)] = 0 $

*Variance* sous $H_0$ : on utilise $bold(w)^top bold(x) tilde cal(N)(0, ||bold(w)||_2^2 sigma_x^2)$ :

$ sigma_0^2 = "Var"(S(bold(y))|H_0) = frac(||bold(w)||_2^2 sigma_x^2, M^2) $

=== Calcul sous $H_1$

*Moyenne* sous $H_1$ : puisque $bold(y) = bold(x) + bold(w)$ :

$ mu_1 = EE[S(bold(y)) | H_1] = 1/M (EE[bold(x)^top bold(w)] + EE[bold(w)^top bold(w)]) = frac(||bold(w)||_2^2, M) $

*Variance* sous $H_1$ : la partie $bold(w)$ est constante, donc :

$ sigma_1^2 = "Var"(S(bold(y))|H_1) = frac(||bold(w)||_2^2 sigma_x^2, M^2) = sigma_0^2 $

=== Résumé des distributions

#grid(
  columns: (1fr, 1fr),
  column-gutter: 1em,
  rect(inset: 14pt, radius: 5pt, fill: rgb("#dbeafe"), stroke: rgb("#3b82f6"))[
    *Sous $H_0$*

    $ S tilde cal(N)(mu_0, sigma_0^2) $

    $ mu_0 = 0 $

    $ sigma_0^2 = frac(||bold(w)||_2^2 sigma_x^2, M^2) $
  ],
  rect(inset: 14pt, radius: 5pt, fill: rgb("#fee2e2"), stroke: rgb("#ef4444"))[
    *Sous $H_1$*

    $ S tilde cal(N)(mu_1, sigma_1^2) $

    $ mu_1 = frac(||bold(w)||_2^2, M) $

    $ sigma_1^2 = frac(||bold(w)||_2^2 sigma_x^2, M^2) $
  ],
)

#insight-box[
  Les deux distributions sont *gaussiennes avec la même variance* mais des *moyennes différentes*. Seul le décalage $mu_1 - mu_0 = ||bold(w)||_2^2 / M > 0$ permet de distinguer $H_0$ de $H_1$. Plus l'énergie du tatouage $||bold(w)||_2^2$ est grande, plus les deux distributions sont séparées.
]

// ================================================================
== Probabilités d'erreur
// ================================================================

=== La fonction Q

#def-box("Fonction Q")[
  $ Q(x) = frac(1, sqrt(2 pi)) integral_x^(+infinity) e^(-t^2\/2) dif t $

  Probabilité de la *queue droite* d'une loi normale standard $cal(N)(0,1)$.
]

Propriétés utiles :
- $Q(0) = 0.5$
- $Q(x) + Q(-x) = 1$
- $Q(x) → 0$ quand $x → +infinity$

```python
from scipy.stats import norm
Q = lambda x: 1 - norm.cdf(x)  # Calcule Q(x)
```

=== Probabilité de fausse acceptation ($P_"FA"$)

*Définition* : probabilité de déclarer $H_1$ alors que $H_0$ est vraie.

$ P_"FA"(gamma) = Pr(S(bold(y)) >= gamma | H_0) = Q lr((frac(gamma - mu_0, sigma_0))) $

Avec nos paramètres :

$ P_"FA"(gamma) = Q lr((frac(gamma, sqrt(||bold(w)||_2^2 sigma_x^2 \/ M^2)))) $

=== Probabilité de détection correcte ($P_D$)

*Définition* : probabilité de déclarer $H_1$ quand $H_1$ est vraie.

$ P_D(gamma) = Pr(S(bold(y)) >= gamma | H_1) = Q lr((frac(gamma - mu_1, sigma_1))) $

Avec nos paramètres :

$ P_D(gamma) = Q lr((frac(gamma - ||bold(w)||_2^2\/M, sqrt(||bold(w)||_2^2 sigma_x^2 \/ M^2)))) $

=== Probabilité de manqué ($P_"miss"$)

$ P_"miss"(gamma) = 1 - P_D(gamma) $

=== Visualisation des distributions et des erreurs

#align(center)[
  #canvas({
    import draw: *

    // === Axes ===
    line((-4.5, 0), (8, 0), mark: (end: ">"), stroke: 1pt)
    line((-4, -0.15), (-4, 2.5), mark: (end: ">"), stroke: 1pt)

    content((8.3, 0), text(size: 9pt)[$S$])
    content((-4.2, 2.6), text(size: 8pt)[Densité])

    // === Courbe H0 (centrée en -1) ===
    // Approximation gaussienne par bezier
    bezier(
      (-4.2, 0.05), (-2.5, 0.05),
      (-3.8, 0.05), (-3.3, 0.1),
      stroke: rgb("#3b82f6") + 2pt
    )
    bezier(
      (-3.3, 0.1), (-2.0, 1.5),
      (-3.0, 0.5), (-2.4, 1.4),
      stroke: rgb("#3b82f6") + 2pt
    )
    bezier(
      (-2.0, 1.5), (-1.0, 2.2),
      (-1.7, 1.9), (-1.3, 2.2),
      stroke: rgb("#3b82f6") + 2pt
    )
    bezier(
      (-1.0, 2.2), (-0.0, 1.5),
      (-0.7, 2.2), (-0.3, 1.9),
      stroke: rgb("#3b82f6") + 2pt
    )
    bezier(
      (-0.0, 1.5), (0.8, 0.1),
      (0.3, 1.4), (0.6, 0.5),
      stroke: rgb("#3b82f6") + 2pt
    )
    bezier(
      (0.8, 0.1), (1.5, 0.05),
      (1.0, 0.1), (1.3, 0.05),
      stroke: rgb("#3b82f6") + 2pt
    )

    content((-1.0, 2.5), text(fill: rgb("#1d4ed8"), weight: "bold")[$H_0$])
    content((-1.0, -0.3), text(size: 9pt, fill: rgb("#1d4ed8"))[$mu_0 = 0$])

    // === Courbe H1 (centrée en 4) ===
    bezier(
      (1.5, 0.05), (2.2, 0.1),
      (1.7, 0.05), (2.0, 0.08),
      stroke: rgb("#ef4444") + 2pt
    )
    bezier(
      (2.2, 0.1), (3.2, 1.5),
      (2.5, 0.5), (2.9, 1.4),
      stroke: rgb("#ef4444") + 2pt
    )
    bezier(
      (3.2, 1.5), (4.0, 2.2),
      (3.5, 1.9), (3.8, 2.2),
      stroke: rgb("#ef4444") + 2pt
    )
    bezier(
      (4.0, 2.2), (5.0, 1.5),
      (4.3, 2.2), (4.7, 1.9),
      stroke: rgb("#ef4444") + 2pt
    )
    bezier(
      (5.0, 1.5), (5.8, 0.1),
      (5.3, 1.4), (5.6, 0.5),
      stroke: rgb("#ef4444") + 2pt
    )
    bezier(
      (5.8, 0.1), (6.8, 0.05),
      (6.0, 0.1), (6.5, 0.05),
      stroke: rgb("#ef4444") + 2pt
    )

    content((4.2, 2.5), text(fill: rgb("#dc2626"), weight: "bold")[$H_1$])
    content((4.0, -0.3), text(size: 9pt, fill: rgb("#dc2626"))[$mu_1$])

    // === Seuil gamma ===
    line((2.5, 0), (2.5, 2.3),
      stroke: (paint: rgb("#374151"), thickness: 1.5pt, dash: "dashed")
    )
    content((2.5, -0.3), text(weight: "bold")[$gamma$])

    // === Régions d'erreur (texte) ===
    content((1.5, 0.7), text(size: 8pt, fill: rgb("#f59e0b"))[Fausse \ acceptation])
    content((3.5, 0.7), text(size: 8pt, fill: rgb("#22c55e"))[Détection \ correcte])

  })
]

// ================================================================
== Courbe ROC (_Receiver Operating Characteristic_)
// ================================================================

=== Définition

La *courbe ROC* représente le compromis entre $P_D(gamma)$ et $P_"FA"(gamma)$ quand le *seuil $gamma$ varie* de $+infinity$ à $-infinity$.

#align(center)[
  #canvas({
    import draw: *

    // === Cadre ===
    rect((0, 0), (5, 5), stroke: (paint: rgb("#d1d5db"), thickness: 1pt))

    // === Axes ===
    line((-0.2, 0), (5.5, 0), mark: (end: ">"), stroke: 1.2pt)
    line((0, -0.2), (0, 5.5), mark: (end: ">"), stroke: 1.2pt)

    content((5.8, 0), text(size: 9pt)[$P_"FA"$])
    content((-0.3, 5.7), text(size: 9pt)[$P_D$])

    // === Diagonale (aléatoire) ===
    line((0, 0), (5, 5),
      stroke: (paint: rgb("#9ca3af"), thickness: 1pt, dash: "dashed")
    )
    content((3.8, 3.0), text(size: 8pt, fill: rgb("#6b7280"))[Décision \ aléatoire])

    // === Courbe ROC ===
    bezier(
      (0, 0), (5, 5),
      (0, 3.5), (2.0, 5.0),
      stroke: rgb("#ef4444") + 2.5pt
    )
    content((0.8, 3.5), text(fill: rgb("#dc2626"), weight: "bold")[Courbe ROC])

    // === Point opérationnel ===
    circle((1.5, 3.8), radius: 0.12, fill: rgb("#dc2626"), stroke: none)
    content((2.3, 3.8), text(size: 9pt)[$gamma$ spécifique])

    // === Idéal ===
    circle((0, 5), radius: 0.15, fill: rgb("#22c55e"), stroke: rgb("#16a34a"))
    content((0.8, 5.2), text(size: 9pt, fill: rgb("#16a34a"))[Idéal])

    // === Graduations ===
    content((-0.35, 5), text(size: 9pt)[1.0])
    content((-0.35, 0), text(size: 9pt)[0.0])
    content((5, -0.3), text(size: 9pt)[1.0])
    content((0, -0.3), text(size: 9pt)[0.0])
  })
]

- *Idéal* : $P_D → 1$ et $P_"FA" → 0$ (coin supérieur gauche)
- *Aléatoire* : diagonale (pire cas)
- *AUC* (_Area Under the Curve_) : mesure globale des performances (1 = parfait, 0.5 = aléatoire)

// ================================================================
== Stratégies de sélection du seuil $gamma$
// ================================================================

=== Stratégie de Neyman-Pearson

#def-box("Neyman-Pearson")[
  *Objectif* : minimiser $P_"miss"(gamma)$ sous une contrainte fixée sur $P_"FA"$ :

  $ max_gamma P_D(gamma) quad "sous la contrainte" quad P_"FA"(gamma) leq alpha $

  Le seuil optimal s'obtient en inversant :
  $ gamma^* = Q^(-1)(alpha) dot sigma_0 + mu_0 $
]

*Usage typique* : en sécurité, on fixe un taux de fausse alarme maximal tolérable (ex. $alpha = 10^{-6}$).

=== Stratégie Bayésienne

#def-box("Critère Bayésien")[
  *Objectif* : minimiser la probabilité d'erreur *totale* pondérée par les probabilités a priori :

  $ P_"erreur"(gamma) = P_"FA"(gamma) dot P(H_0) + P_"miss"(gamma) dot P(H_1) $

  *Cas symétrique* ($P(H_0) = P(H_1) = 0.5$) :
  $ P_"erreur"(gamma) = 1/2 (P_"FA"(gamma) + P_"miss"(gamma)) $

  *Seuil optimal* pour deux gaussiennes de même variance :
  $ gamma_"opt" = frac(mu_0 + mu_1, 2) $
]

=== Comparaison des deux approches

#grid(
  columns: (1fr, 1fr),
  column-gutter: 1em,
  rect(inset: 12pt, radius: 5pt, fill: rgb("#dbeafe"), stroke: rgb("#3b82f6"))[
    *Neyman-Pearson*

    ✅ Contrôle exact du $P_"FA"$

    ✅ Maximise $P_D$ pour un $P_"FA"$ donné

    ✅ Préféré en sécurité (peu de fausses alarmes)

    ❌ Ignore les probabilités a priori
  ],
  rect(inset: 12pt, radius: 5pt, fill: rgb("#f0fdf4"), stroke: rgb("#22c55e"))[
    *Bayésien*

    ✅ Minimise l'erreur globale

    ✅ Intègre les probabilités a priori

    ✅ Simple à calculer (seuil = moyenne des moyennes)

    ❌ Nécessite de connaître $P(H_0)$ et $P(H_1)$
  ],
)

// ================================================================
= Résumé et Points Clés
// ================================================================

== Notations à retenir absolument

#grid(
  columns: (0.35fr, 0.65fr),
  column-gutter: 1.5em,
  row-gutter: 0.8em,
  align(center + horizon)[$bold(x) in RR^(H times W times C)$], [Image originale RGB],
  align(center + horizon)[$bold(y) = phi(bold(x), bold(k), bold(m))$], [Image tatouée (_stégo_)],
  align(center + horizon)[$bold(w)$, $||bold(w)||_2^2 = E_w$], [Tatouage de référence (énergie $E_w$)],
  align(center + horizon)[$bold(m) in {0,1}^L$], [Message à intégrer ($L$ bits)],
  align(center + horizon)[$S = (bold(y)^top bold(w))/M$], [Statistique de test (score)],
)

== Métriques à connaître

#grid(
  columns: (1fr, 1fr),
  column-gutter: 1em,
  rect(inset: 10pt, radius: 4pt, fill: rgb("#eff6ff"), stroke: rgb("#93c5fd"))[
    *Qualité visuelle*
    - $"MSE"(bold(x), bold(y))$ : distorsion pixel
    - $"PSNR"$ : rapport signal/bruit (dB)
    - $"WMSE"$, $"WPSNR"$ : versions pondérées
  ],
  rect(inset: 10pt, radius: 4pt, fill: rgb("#f0fdf4"), stroke: rgb("#86efac"))[
    *Similarité vectorielle*
    - Produit scalaire $bold(x)^top bold(y)$ : corrélation
    - $cos(theta)$ : similarité directionnelle
    - $||bold(x) - bold(y)||_2$ : distance euclidienne
  ],
)

== Transformées : synthèse

#grid(
  columns: (1fr, 1fr, 1fr, 1fr),
  column-gutter: 6pt,
  rect(inset: 8pt, radius: 4pt, fill: rgb("#fff7ed"), stroke: rgb("#f59e0b"), align: center)[
    *Bit-plane*
    Décompo. binaire
    8 plans par image
  ],
  rect(inset: 8pt, radius: 4pt, fill: rgb("#dbeafe"), stroke: rgb("#3b82f6"), align: center)[
    *DFT / DCT*
    Fréquentiel
    Analytique
  ],
  rect(inset: 8pt, radius: 4pt, fill: rgb("#dcfce7"), stroke: rgb("#22c55e"), align: center)[
    *DWT*
    Multi-résolution
    Spatio-fréquentiel
  ],
  rect(inset: 8pt, radius: 4pt, fill: rgb("#fdf4ff"), stroke: rgb("#d8b4fe"), align: center)[
    *PCA / DL*
    Apprenante
    Non-linéaire
  ],
)

== Détection : synthèse

#grid(
  columns: (0.45fr, 0.55fr),
  column-gutter: 1em,
  row-gutter: 0.7em,
  [*Score $S$*], [$S = bold(y)^top bold(w) / M$ — produit scalaire normalisé],
  [*Sous $H_0$*], [$S tilde cal(N)(0, sigma_0^2)$ avec $sigma_0^2 = ||bold(w)||_2^2 sigma_x^2 / M^2$],
  [*Sous $H_1$*], [$S tilde cal(N)(mu_1, sigma_1^2)$ avec $mu_1 = ||bold(w)||_2^2 / M$],
  [*$P_"FA"$*], [$Q((gamma - mu_0)/sigma_0)$ — probabilité fausse alarme],
  [*$P_D$*], [$Q((gamma - mu_1)/sigma_1)$ — probabilité de détection],
  [*Courbe ROC*], [Compromis $P_D$ vs $P_"FA"$ en fonction de $gamma$],
)

== Pour aller plus loin

#attention-box[
  *Prochaine séance* : exploration détaillée des *techniques de tatouage numérique*, en utilisant toutes les notations introduites ici. Assurez-vous de maîtriser la formule du score $S$ et les distributions sous $H_0$ / $H_1$ — elles sont le *fondement de toute la suite du cours*.
]

*Ressources recommandées* :

- Revoir les propriétés des distributions gaussiennes
- Revoir la notion de test d'hypothèses (Neyman-Pearson)
- Pratiquer le calcul de MSE et PSNR en Python (`numpy`, `skimage.metrics`)
- Manipuler les transformées DFT et DCT en Python (`numpy.fft`, `scipy.fft`)
