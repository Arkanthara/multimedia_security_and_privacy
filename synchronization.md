Je veux que tu réimplémentes le code sync.py (tu peux changer tout, même les fonctions).

Actuellement, `sync.py` utilise l'auto-corrélation puis la phase corrélation dans l'espace log-polaire pour estimer la rotation et l'échelle, puis la phase corrélation dans l'espace cartésien pour estimer la translation.

Au lieu d'utiliser l'auto-corrélation, je veux que tu utilises la Fourier-Mellin transform avec phase correlation pour estimer la rotation et l'échelle, puis la phase corrélation dans l'espace cartésien pour estimer la translation.

Donc en gros, tu vas faire :
1. FFT de l'image et de la référence construite
2. Calcul de la magnitude des FFTs
3. Remove DC component (mettre à zéro la composante à basse fréquence) (si ça aide à la robustesse)
4. Logarithmiser les magnitudes (si ça aide à la robustesse)
5. Normaliser les magnitudes (si ça aide à la robustesse)
6. Resample en coordonnées log-polaire
7. Phase correlation pour estimer rotation et échelle

Tu peux utiliser OpenCV ou scikit-image pour les transformations et les corrélations.
Assure-toi que le code est clair, bien structuré et commenté.
Je veux que le résultat soit très robuste à la rotation, à l'échelle et à la translation.

Pour la correction de l'image, tu dois utiliser un processus similaire à celui qui existe déjà, mais tu peux l'améliorer si tu penses que c'est nécessaire.