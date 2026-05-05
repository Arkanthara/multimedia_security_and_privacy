Salut !!!
Tu es un expert en traitement d'images et en watermarking, et tu as une grande expérience dans l'implémentation de techniques de watermarking robustes et imperceptibles en utilisant Python et des bibliothèques telles que scikit-image, scipy et numpy (et OpenCV).
Je veux que tu implémentes un **watermarking utilisant la décomposition en valeurs singulières (SVD)**, qui est une technique de watermarking dans le domaine spatial.

Voici les étapes que tu dois suivre pour l'implémenter:
- Préprocessing du watermark
    - Si le watermark doit être répété, faire en sorte que ce soit le cas.
    - Retourner le watermark préprocessé
- Décomposition en valeurs singulières de l'image d'origine (U, S, V)
- Embedding du watermark dans les valeurs singulières S:
    - Générer pseudo-aléatoirement des positions dans S pour insérer les bits du watermark (utiliser la clé pour la reproductibilité)
    - Insérer les bits du watermark dans S à ces positions.
        - Insertion utilisant une technique de quantification (QIM):
            - Si le bit à insérer est 1, quantifier la valeur singulière à l'intervalle [delta/2, delta] mod delta
            - Si le bit à insérer est 0, quantifier la valeur singulière à l'intervalle [0, delta/2] mod delta
            - delta est un paramètre de quantification à déterminer expérimentalement pour trouver le meilleur compromis entre robustesse et imperceptibilité
            - Attention à ne pas changer l'ordre des valeurs singulières, sinon la reconstruction de l'image sera très dégradée
        - Pour l'insertion, comme les valeurs singulières sont ordonnées de manière décroissante exponentiellement, je veux que delta soit appliqué, si activé, pour que à partir du moment où par exemple la ième valeur singulière est égale à la première valeur singulière multipliée par k^i, alors delta devient delta * k^i, pour que les modifications soient plus importantes sur les premières valeurs singulières et moins importantes sur les dernières, afin de préserver la qualité de l'image watermarquée.
- Reconstruction de l'image watermarquée à partir de U, S modifié et V
- Retourner l'image watermarquée

Pour le décodage:
- Décomposition en valeurs singulières de l'image à décoder (U', S', V')
- Extraction des bits du watermark à partir de S':
    - Utiliser les mêmes positions pseudo-aléatoires dans S' pour extraire les bits du watermark
    - Pour chaque position, déterminer si le bit est 1 ou 0 en fonction de la quantification de la valeur singulière à cette position (utiliser les mêmes intervalles que pour l'insertion... attention à appliquer le même delta que pour l'insertion, en fonction de la position dans S')
- Post-processing du watermark extrait (par exemple, si répétition du watermark, faire un vote majoritaire pour déterminer les bits du watermark final)

La classe WatermarkModel doit accepter les paramètres suivants:
- psnr_threshold: float = 30.0,
- max_encode_time: float = 5.0,
- max_decode_time: float = 1.0, (ces premiers paramètres ne te sont pas utiles, mais doivent être présents dans la classe pour être compatibles avec les autres modèles)
- message_length: int = 32,
- message_repetitions: int = 1,
- quantization_delta: int = 10,
- decay_factor: float = 1.0 (facteur de décroissance pour l'application de delta sur les valeurs singulières),
- key: int = 42 (clé utilisée pour toutes les générations aléatoires pour la reproductibilité)

Le code doit être:
- Minimal, optimisé (utilisant les fonctions existantes et numpy vectorization), clair, bien structuré et commenté, documenté en style numpy.
- Tu peux utiliser OpenCV ou scikit-image selon tes besoins, évite de réimplémenter des fonctions déjà existantes dans ces bibliothèques.
- Code, documentation et commentaires doivent être en anglais. Je veux qu'il soit facile à comprendre et à maintenir.
