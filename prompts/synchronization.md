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


Je veux maintenant que tu modifies 'model.py' (actuellement, LSB watermarking) pour faire du watermarking spatial (Garde la classe et les 2 méthodes encode et decode).
Tu peux modifier patch.py selon tes besoins. (besoin d'updater patch en -1, +1 pour le watermarking spatial par exemple, ou l'extraction des bits du watermark pour le decoding...)

La méthode de watermarking spatial que je veux que tu implémentes est la suivante:

Encode: 

- Préprocessing du watermark
    - Si le watermark doit être répété, faire en sorte que ce soit le cas.
    - Si un code de correction d'erreur doit être utilisé, l'appliquer au watermark. (Note: le nombre de répétitions du watermark et du code de correction d'erreur sont à déterminer expérimentalement pour trouver le meilleur compromis entre robustesse et imperceptibilité)
    - Retourner le watermark préprocessé
- Embedding du watermark dans le patch et padding du patch pour qu'il ait la même taille que l'image d'origine. (utilise les fonctions existantes de patch.py pour ça !!! Note: défaut est mode symétrique)
- Convert patch en -1, + 1 (au lieu de 0, 1)
- Compute la NVF (noise visibility function) de l'image d'origine (instructions ci-dessous)
- Embedding du watermark: img = img + alpha_1 * NVF * w + alpha_2 * (1 - NVF) * w

Decoding:

- Create reference patch (utiliser les fonctions existantes de patch.py pour ça)
- Calculer la NVF de l'image d'origine
- Creer reference avec alpha_1 * NVF * ref + alpha_2 * (1 - NVF) * ref
- Denoise image (utiliser wiener filter avec noise variance estimée à partir de la ref)
- image - denoised image
- synchronisation
- extraction du watermark (La synchronisation renvoie déjà un patch aligné avec la référence contenant l'addition de tous les blocks, donc tu peux directement procéder à l'extraction du watermark: > 0 1, sinon 0)
- Post-processing du watermark extrait (par exemple, si un code de correction d'erreur a été utilisé, l'appliquer pour corriger les erreurs dans le watermark extrait, ou si répétition du watermark, faire un vote majoritaire pour déterminer les bits du watermark final)

La classe doit accepter les paramètres suivants:
- alpha_1: poids du watermark dans les zones à forte variance
- alpha_2: poids du watermark dans les zones à faible variance
- D: paramètre de contrôle de la NVF
- use_nvf: booléen indiquant si la NVF doit être utilisée pour le watermarking (si False, alors img = img + alpha_1 * w)
- nvf_window_size: taille de la fenêtre utilisée pour calculer la NVF
- msg_length: longueur du message à encoder dans le watermark
- use_ecc: booléen indiquant si un code de correction d'erreur doit être utilisé
- ecc_repetitions: nombre de répétitions du code de correction d'erreur
- msg_repetitions: nombre de répétitions du watermark
- patch_size: taille des patches utilisés pour l'embedding et la synchronisation
- upsampling_factor: facteur d'upsampling pour les patches
- key: clé utilisée pour toutes les générations aléatoires pour la reproductibilité



NVF: 

local_variance = compute_local_variance(image , WINDOW_SIZE)
max_variance = np. max (local_variance)
nvf = 1 / (1 + D * local_variance / max_variance)
Les paramètres alpha_1, alpha_2 et D et nvf_window_size sont à déterminer expérimentalement pour trouver le meilleur compromis entre robustesse et imperceptibilité.

Error correction code:

Tu dois créer un nouveau module `error_correction.py` dans utils qui implémente des utilitaires pour les codes de correction d'erreur.
Je veux que tu utilises la librairie `ldpc` pour utiliser des codes LDPC (Low-Density Parity-Check) pour la correction d'erreur.

Le code doit être minimal, optimisé (utilisant les fonctions existantes et numpy vectorization), clair, bien structuré et commenté, documenté en style numpy. Tu peux utiliser OpenCV ou scikit-image selon tes besoins, évite de réimplémenter des fonctions déjà existantes dans ces bibliothèques.

Code, documentation et commentaires doivent être en anglais. Je veux qu'il soit facile à comprendre et à maintenir.