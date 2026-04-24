Voici un code sync.py.

Dans ce code, pour estimer la transformation affine, il faut détecter les points d'intérêt dans les deux images, puis trouver les correspondances entre ces points. Ensuite, on peut utiliser ces correspondances pour calculer la transformation affine qui aligne les deux images.

Je veux que tu travailles sur la partie de détection des points d'intérêt et de correspondance.
Voici comment procéder :

- Extraire les coordonnées du centre de l'image (cx, cy) (pixel non nul vers le centre de l'image)
- Extraire les coordonnées des pixels non nuls après avoir appliqué non_max_suppression.
- Centrer les coordonnées des points d'intérêt en soustrayant (cx, cy) de chaque point.
- Prendre le point qui a la plus petite distance euclidienne au centre de l'image.
- Prendre le deuxième point qui a la plus petite distance euclidienne au premier point (qui ne soit pas le centre de l'image).
- Prendre le troisième point qui a la plus petite distance euclidienne au deuxième point (qui ne soit pas le centre de l'image).
- Ces points dans l'ordre (premier, deuxième, troisième) seront utilisés pour estimer la transformation affine.
- Faire la même chose pour l'image et la référence
- Estimer la transformation affine à partir des correspondances des points d'intérêt entre les deux images.

Je veux le code le plus simple et optimisé possible, court, documenté utilisant numpy style, lisible et maintenable.
Je veux que tu supprimes les fonctions qui ne servent à rien.
Tout les commentaires et documentation et code doivent etre faits en anglais.