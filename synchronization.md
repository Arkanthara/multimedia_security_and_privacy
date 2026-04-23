Je veux que tu réimplémentes le code sync.py (tu peux changer tout, même les fonctions... L'output doit être un block contenant l'addition de tous les blocks, mais aligné avec la référence)...
Tu peux aussi apporter quelques changements à patch.py (comme génération de patch en -1, +1 au lieu de 0, 1) si tu penses que c'est nécessaire pour la synchronisation... Mais l'essentiel est de faire une nouvelle implémentation de la synchronisation qui est plus robuste à la rotation et à l'échelle, et qui utilise des pics d'auto-corrélation pour estimer la transformation affine, puis une corrélation pour estimer la translation.

Actuellement, `sync.py` utilise l'auto-corrélation puis la phase corrélation dans l'espace log-polaire pour estimer la rotation et l'échelle, puis la phase corrélation dans l'espace cartésien pour estimer la translation.

A la place, je veux que tu utilises cette approche détaillée ci-dessous, qui est plus robuste à la rotation et à l'échelle, et qui utilise des pics d'auto-corrélation pour estimer la transformation affine, puis une corrélation pour estimer la translation.
Synchronisation:
Pour affine transformation:
- Auto-corrélation image dans fourier (F_img * F_img.conj())
(def autocorrelation_fft(image):
f = np.fft.fft2(image)
f_conj = np.conj(f)
return np.fft.fftshift(np.real(np.fft.ifft2(f * f_conj))))
- Auto-corrélation référence dans fourier (F_ref * F_ref.conj())
- Appliquer la NMS (non-maximum suppression) pour trouver les pics dans les auto-corrélations
(def non_maximum_suppression(image , size=31):
max_filtered = maximum_filter(image , size=size)
nms_result = (image == max_filtered) * image # Retain only local maxima
return nms_result)
- Trouver le pic central dans l'auto-corrélation de référence (normalement, grâce à fftshift, il devrait être au centre de l'image)
- Le centre est le même pour les deux auto-corrélations.
- Trouver les 2 autres pics les plus proches du centre dans l'auto-corrélation de l'image (correspondant aux pics sur les axes x et y dans l'auto-corrélation de référence... Applique donc une recherche seulement dans la région supérieure du centre...)
- Trouver Le pic qui est le plus proche des 2 pics trouvés et du centre.
- Extraire les coordonnées des pics trouvés dans l'image et dans la référence
- Calculer la transformation affine permettant de faire correspondre les pics de l'image avec les pics de la référence
- Appliquer la transformation à l'image pour la synchroniser avec la référence

Pour la translation: 
- Diviser l'image synchronisée en blocks (même taille que les patches) et les additionner pour faire un unique block (note que pour les additions, tu dois faire attention au symmétrique padding pour que les blocks ne soient pas flippés à cause du padding)
- Appliquer corrélation entre ce block et un block de référence (tu peux facilement en construire un à partir du code dans patch.py)
- Appliquer la NMS pour trouver le pic de corrélation
- Extraire les coordonnées du pic de corrélation
- Calculer la translation à appliquer pour réaligner l'image avec la référence (pic de corrélation - centre du block)
- Appliquer la translation à l'image pour la synchroniser avec la référence (utiliser par exemple un roll pour faire une translation circulaire)
- Retourner le résultat synchronisé (c'est donc un patch contenant l'addition de tous les blocks, mais aligné avec la référence)

Chaque fois que tu as besoin de faire par exemple une transformation affine qui requiert un padding, tu dois faire un crop pour éviter tout padding qui pourrait introduire des artefacts, comme dans la version actuelle de sync.py.


Tu peux utiliser OpenCV ou scikit-image pour les transformations et les corrélations.
Assure-toi que le code est clair, bien structuré, commenté utilisant numpy style, aussi simple que possible et optimisé.