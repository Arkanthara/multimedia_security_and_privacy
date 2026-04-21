Tu es un ingénieur Python senior.

Ta mission est d’implémenter un **LSB watermarking simple, vectorisé et déterministe**, basé sur :

* un **patch pseudo-aléatoire structuré**
* une **synchronisation robuste en domaine fréquentiel**

Le code doit être clair, court, modulaire et prêt à être amélioré.

---

# STRUCTURE

* `lsb/model.py`
* `utils/patch.py`
* `utils/sync.py`

---

# PIPELINE GLOBAL

## ENCODE

1. Générer un patch binaire pseudo-aléatoire avec une clé
2. Générer des positions pour les bits dans ce patch
3. Insérer les bits du watermark dans le patch
4. Upsample le patch (x2) pour robustesse
5. Tile le patch sur toute l’image :

   * mode "normal"
   * mode "symmetric" (ex: `np.pad(..., mode="symmetric")`)
6. Encoder dans le LSB

---

## DECODE

1. Extraire le LSB

2. Reconstruire le patch attendu avec la clé

3. Si `sync_enabled=True` :

   * **Étape 1 (rapide)** : tenter un décodage direct (sans correction).
     → si la **confidence ≥ 0.8**, retourner immédiatement.
   * **Étape 2 (rotation + scale)** :

     * estimer rotation et scale via la sync
     * réaligner l’image
     * décoder → si confidence ≥ 0.8, retourner
   * **Étape 3 (translation fine)** :

     * estimer translation via phase correlation
     * réaligner l'image
     * décoder
   * **Retourner le meilleur résultat** (plus haute confidence)

4. Si `sync_enabled=False` :

   * décodage direct uniquement

5. Lecture des bits :

   * utiliser les positions déterministes du patch

6. Majority voting :

   * convertir LSB en `{-1, +1}`
   * sommer les contributions de tous les patches
   * signe → bit (positif = 1, négatif = 0)
   * **confidence** = moyenne des valeurs absolues normalisées par bit
     (doit être ≥ 0.8 pour être considéré fiable)

7. Retourner le meilleur watermark trouvé

---

# PATCH — `utils/patch.py`

Implémenter une logique **déterministe, inversible et centralisée** :

### 1. Génération du patch

* Patch binaire pseudo-aléatoire `(patch_size, patch_size)` via RNG(seed=key)

### 2. Positions des bits

* Positions pseudo-aléatoires dans le patch
* Déterministes (même key → mêmes positions)

### 3. Embedding du watermark

* Injection des bits aux positions
* Gestion de `repeat`

### 4. Upsampling

* x2 via répétition (chaque pixel → bloc 2x2)

### 5. Tiling (CRITIQUE)

* Adapter à la taille image
* Deux modes :

  * **normal** : tile + crop
  * **symmetric** : padding symétrique
* Le patch **doit commencer en (0,0) avec sa valeur présente à la position (0, 0) → crucial pour la sync**

### 6. Extraction

* Lire aux mêmes positions
* Conversion en `{-1, +1}` pour le voting

### 7. Majority voting

* Somme vectorisée
* Décision par signe

---

# SYNCHRONISATION — `utils/sync.py`

Objectif : **réaligner l’image sur la structure du patch**

### Pipeline Decode

1. Extraire LSB → image binaire

2. Autocorrélation (image) :

   * FFT(image)
   * multiplier par conjugué
   * IFFT → autocorrélation
   * conversion log-polar

3. Générer la **structure de référence** :

   * reconstruire le patch via la clé (sans watermark)
   * même pipeline (tiling + upsample)

4. Autocorrélation (référence) :

   * même processus FFT → log-polar

5. Phase correlation (log-polar) :

   * estimer **rotation + scale**

6. Correction rotation/scale :

   * appliquer transformation inverse
   * **autoriser crop (ne pas forcer resize)** pour éviter artefacts

7. Régénérer la structure à la nouvelle taille

8. Phase correlation (cartésien) :

   * entre image corrigée et structure
   * estimer translation

9. Correction translation + crop final

10. Régénérer le patch final aligné

### Contraintes

* déterministe
* vectorisé
* pipeline clair
* éviter toute complexité inutile

---

# LSB — `lsb/model.py`

### Encode

* convertir image → uint8
* générer patch final (taille image)
* remplacer LSB :

```python
image = (image & ~1) | patch
```

---

### Decode

* extraire LSB
* si activé → synchronisation complète (rotation/scale → translation)
* extraire bits via `utils.patch`
* majority voting
* early exit si confidence ≥ 0.8

---

# CONTRAINTES

* code court, lisible
* déterminisme strict
* patch uniquement dans `utils/patch.py`
* sync uniquement dans `utils/sync.py`
* aucune duplication
* vectorisation maximale
*  

  code documenté EN ANGLAIS (style numpy pour les fonctions)

    
* Nom de fonctions et variables "meaningfull in english"

---

# IMPORTANT

* le patch doit être **reconstructible uniquement avec la clé**
* la synchronisation doit permettre un **réalignement robuste après transformations**

---

# LIVRABLE

* `utils/patch.py`
* `utils/sync.py`
* `lsb/model.py`
