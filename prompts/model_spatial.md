Je veux maintenant que tu fasses une nouvelle implémentation du modèle pour non pas travailler avec LSB, mais avec l'image dans le domaine spatial.

Et je veux surtout que tu implémentes une nouvelle version pour la synchronisation, qui est plus robuste à la rotation et à l'échelle (note que le patch doit avoir des valeurs -1 et +1 pour faire une cross corrélation correcte ).

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



Encode: 

- Embedding du watermark dans le patch et padding du patch pour qu'il ait la même taille que l'image d'origine. (utilise les fonctions existantes de patch.py pour ça)
- Convert patch en -1, + 1 (au lieu de 0, 1)
- Compute la NVF (noise visibility function) de l'image d'origine
- Embedding du watermark: img = img + alpha_1 * NVF * w + alpha_2 * (1 - NVF) * w

Decoding:

- Create reference patch (utiliser les fonctions existantes de patch.py pour ça)
- Calculer la NVF de l'image d'origine
- Creer reference avec alpha_1 * NVF * ref + alpha_2 * (1 - NVF) * ref
- Denoise image (utiliser wiener filter avec noise variance estimée à partir de la ref)
- image - denoised image
- synchronisation
- extraction du watermark (utiliser les fonctions existantes de patch.py pour ça... Note que le denoised image est deja en -alpha, +beta...)

NVF: 

local_variance = compute_local_variance(image , WINDOW_SIZE)
max_variance = np. max (local_variance)
nvf = 1 / (1 + D * local_variance / max_variance)
Les paramètres alpha_1, alpha_2 et D et nvf_window_size sont à déterminer expérimentalement pour trouver le meilleur compromis entre robustesse et imperceptibilité.

Le code doit être minimal, optimisé (utilisant les fonctions existantes et numpy vectorization), clair, bien structuré et commenté, documenté en style numpy. Tu peux utiliser OpenCV ou scikit-image selon tes besoins, évite de réimplémenter des fonctions déjà existantes dans ces bibliothèques.

Code, documentation et commentaires doivent être en anglais. Je veux qu'il soit facile à comprendre et à maintenir.
