#show figure.where(kind: "subfigure"): set figure(supplement: "Figure")

#show figure.where(kind: image): outer => {
  counter(figure.where(kind: "subfigure")).update(0)
  set figure(numbering: (..nums) => {
    let outer-nums = counter(figure.where(kind: image)).at(outer.location())
    std.numbering("1a", ..outer-nums, ..nums)
  })
  show figure.where(kind: "subfigure"): inner => {
    show figure.caption: it => context {
      std.numbering("(a)", it.counter.at(inner.location()).last())
      [ ]
      it.body
    }
    inner
  }
  outer
}
// Main report file

#import "@preview/touying:0.7.3": *
#import themes.university: *
#import themes.stargazer: *
#import "@preview/chronos:0.3.0": *
#import "@preview/pintorita:0.1.4"
#import "@preview/lilaq:0.6.0" as lq
#import "@preview/tiptoe:0.4.0"


#import "@preview/theorion:0.6.0": *
#import cosmos.clouds: *
#show: show-theorion

#show raw.where(lang: "pintora"): it => pintorita.render(it.text)

#import "@preview/numbly:0.1.0": numbly

// #context pdfpc.pdfpc-file(here())

#show: stargazer-theme.with(
  aspect-ratio: "16-9",
  // Fix header logo: box it with explicit height and vertical alignment  
  header-right: self => {  
    box(utils.display-current-heading(level: 1))  
    h(.2em)  
    box(height: 1.5em, baseline: 30%, image("img/unige.svg", height: 1.5em))  
  },  
  config-info(
    title: [Multimedia Security and Privacy],
    subtitle: [],
    author: [Michel Jean Joseph Donnet],
    date: datetime.today(),
    institution: [Faculty of Science, University of Geneva],
    contact: [],
    logo: image("./img/unige.svg", height: 2em),
  ),
    // transparence du décor de fond
  alpha: 12%,

  // barre de progression discrète
  progress-bar: true,

  // Palette personnalisée
  config-colors(
    // Couleur institutionnelle adoucie
    // primary: rgb("#CF0063"),
    primary: rgb("#0A7A66"),

    // Version sombre pour slides focus / contrastes
    // primary-dark: rgb("#7A0042"),
    primary-dark: rgb("#065345"),

    // Couleur du texte sur fonds colorés
    secondary: rgb("#FFFFFF"),

    // Couleur faculté (vert sarcelle)
    tertiary: rgb("#0A7A66"),

    // Fond général légèrement cassé
    neutral-lightest: rgb("#FAFAFA"),

    // Texte principal
    // neutral-darkest: rgb("#7A0042"),
    neutral-darkest: rgb("#032e26"),
  ),
)


// #set heading(numbering: numbly("{1}.", default: "1.1"))

#show raw.where(block: true): set block(fill: luma(240), inset: 1em, radius: 0.5em, width: 100%)
// #show raw.where(block: false): set block(fill: luma(240), inset: 1em, radius: 0.5em, width: 100%)
// #show raw.where(block: false): box.with(
//   fill: rgb("#e573e927"),
//   inset: (x: 3pt, y: 0pt),
//   outset: (y: 3pt),
//   radius: 2pt,
// )

// #import "graph_utils.typ": *
#import "neural-viz/lib.typ": *
#import emoji: camera

#import "@preview/algorithmic:1.0.7"
#import algorithmic: style-algorithm, algorithm-figure

#show: style-algorithm

#show raw.where(block: true): set block(fill: luma(240), inset: 1em, radius: 0.5em, width: 100%)

#let src_dir = "../../.."

#title-slide()

#outline-slide()

= Usages of Digital Watermarking

== Watermarking

#tblock(title: "Definition")[
Ensures ownership and authenticity of digital content
]

#tblock(title: "Requirements")[
- Robustness
- Percebility
- Embedding Capacity ($~$64 - 128 bits)
]

#tblock(title: "Applications")[
  - Copyright protection
  - Real vs synthetic media detection
]



== Steganography

#tblock(title: "Definition")[
Hides the existence of a message for secret communication
]

#tblock(title: "Requirements")[
- Undetectability
- Embedding Capacity ($~$1 Ko per image)
]

#tblock(title: "Applications")[
  - Alternative to cryptography
]


== Tamper-proofing

#tblock(title: "Definition")[
Detects unauthorized modifications to digital content (Key based or compression-based)
]

#tblock(title: "Requirements")[
- Capacity
- Robustness
- Localization
]

#tblock(title: "Applications")[
  - Integrity verification
  - Forensics
]

= Watermarking vs Data Hiding

#tblock(title: "Technologies")[
  - Zero-bit watermarking (Detection problem)
  - Multi-bit watermarking (Decoding problem)
]

== Block diagram



#figure(
  {
  let size = (4, 5)
  let nodes = (
    image-node("input", src: src_dir + "/img/tangled_2.png", cover: true, title: "", image-size: size, pos: explicit-pos(0, y: 0)),

    module("embedder", title: [Embedder], pos: right-of("input", by: 2)),
    module("encoder", title: [Encoder], pos: above("embedder")),
    group("embedding-encoder", ("embedder", "encoder"), title: ""),

    text-node("m", [Message *$m$*], pos: above("encoder")),
    text-node("key_1", [#emoji.key], pos: below("embedder")),

    image-node("embed", src: src_dir + "/img/tangled_2.png", cover: true, title: "", image-size: size, pos: right-of("embedder")),
    module("public", title: [Public \ multimedia \ channels], pos: right-of("embed")),
    image-node("distorted", src: src_dir + "/img/attacked.png", cover: true, title: "", image-size: size, pos: right-of("public")),

    module("extractor", title: [Extractor], pos: right-of("distorted")),
    module("decoder", title: [Decoder], pos: above("extractor")),
    group("extractor-decoder", ("extractor", "decoder"), title: ""),

    text-node("hat-m", [Message *$hat(m)$*], pos: above("decoder")),
    text-node("key_2", [#emoji.key], pos: below("extractor")),

    text-node("optional", [*$x$*], pos: right-of("extractor")),

  )

  let edges = (
    ml-edge("input", "embedder", label: [*$x$*], label-side: left, label-pos: 25%),
    ml-edge("input", "encoder", mark: "-|>", from-side: "right", to-side: "left", via: ((1, 0), (1, -1))),
    ml-edge("embedder", "embed", label: [*$y$*]),
    ml-edge("embed", "public"),
    ml-edge("public", "distorted"),
    ml-edge("distorted", "extractor", label: [*$hat(y)$*]),

    ml-edge("m", "encoder"),
    ml-edge("encoder", "embedder"),

    ml-edge("extractor", "decoder"),
    ml-edge("decoder", "hat-m"),

    ml-edge("key_1", "embedder", label: [*$k$*]),
    ml-edge("key_2", "extractor", label: [*$k$*]),

    ml-edge("optional", "extractor", mark: "--|>"),


  )

  ml-diagram(nodes, edges: edges, label-size: 1em, spacing: 1.6em)
  },
) 

= Watermark detection

#tblock(title: "Hypothesis testing")[
  - $H_0: y = x$
  - $H_1: y = x + w$
]

#tblock(title: "Score definition")[
  - $S = (y^T w)/M$ ($M$: number of pixels)
  - $S ~ cal(N)(mu_(H_0), sigma_(H_0)^2)$ under $H_0$ (Central Limit Theorem)
  - $S ~ cal(N)(mu_(H_1), sigma_(H_1)^2)$ under $H_1$
]

#tblock(title: "Probabilities")[
  - Correct Detection: $P_D(gamma) = Pr(S >= gamma | H_1)$
  - False Acceptance: $P_(F A)(gamma) = Pr(S >= gamma | H_0)$
]

== Graph hypothesis testing

#let mu1 = 0
#let sigma1 = 1
#let mu2 = 3
#let sigma2 = 1
#let threshold = 1.2

#let xs = lq.linspace(-5, 8, num: 200)
#let xfa = lq.linspace(threshold, 8, num: 200)
#let xmiss = lq.linspace(-5, threshold, num: 200)

#lq.diagram(
  grid: none,
  width: 25cm,
  height: 10cm,
  bounds: "strict",
  lq.plot(
    label: [$H_0$],
    xs,
    xs.map(x =>
      0.4*calc.exp(-(x - mu1)*(x - mu1) / (2 * sigma1*sigma1))
    ),
    mark: none,
    stroke: 0.2em
  ),
    lq.plot(
    label: [$H_1$],
    xs,
    xs.map(x =>
      0.6*calc.exp(-(x - mu2)*(x - mu2) / (2 * sigma2*sigma2))
    ),
    mark: none,
    stroke: 0.2em
  ),

  lq.fill-between(
    label: [$P_(F A)$],
    xfa,
    xfa.map(x =>
      0.4*calc.exp(-(x - mu1)*(x - mu1) / (2 * sigma1*sigma1))
    ),
    fill: rgb(100%, 5%, 5%, 50%),
  ),

  lq.fill-between(
    label: [$P_(text("miss"))$],
    xmiss,
    xmiss.map(x =>
      0.6*calc.exp(-(x - mu2)*(x - mu2) / (2 * sigma2*sigma2))
    ),
    fill: rgb("#0dffcb80"),
  ),

  lq.vlines(threshold, stroke: (paint: green, thickness: 0.1em), label: [$gamma$])
)

#tblock(title: "Neyman Pearson Strategy")[
  - Fix $P_(F A)$: $ P_(F A)(gamma) = integral_(> gamma) p(x | H_0) d x $
  - Minimize probability of miss: $ P_(text("miss")) = 1 - P_D = integral_(<= gamma) p(x | H_1) d x $
]

#tblock(title: "Bayesian Strategy")[
  - Fix priors $P(H_0)$ and $P(H_1)$
  - Minimize overall error: $P_(text("error")) = P_(F A) P(H_0) + P_(text("miss")) P(H_1)$
]

= Watermark detection in practice

#tblock(title: "Cosine similarity")[
  - $cos(theta) = (y^T w) / (||y||_2 ||w||_2)$
]

#tblock(title: "Score definition")[
  - $S(y) = 1/M (y^T w) = 1/M ||y||_2 ||w||_2 cos(theta)$
]
#tblock(title: [Distribution under $H_0$])[
  - $S(y) = 1/M x^T w ~ cal(N)(0, (||w||_2^2 sigma_x^2)/M^2)$

]

#tblock(title: [Distribution under $H_1$])[
  - $S(y) = 1/M (x + w)^T w ~ cal(N)((||w||_2^2)/M, (||w||_2^2 sigma_x^2)/M^2)$
]

== Decision rule: $|y^T w| > ||y||_2 ||w||_2 cos(theta)$

#lq.diagram(
  grid: none,
  width: 25cm,
  height: 10cm,
  bounds: "strict",
  xaxis: none,
  yaxis: none,
  aspect-ratio: 1,
  lq.ellipse(-1, -1, width: 2, height: 2, stroke: 0.1em),
  lq.line(
    tip: tiptoe.stealth,
    toe: tiptoe.circle,
    (0, 0), (calc.cos(0deg), calc.sin(0deg)),
    stroke: (paint: green, thickness: 0.1em),
    label: [$w$]
  ),
  lq.line(
    tip: tiptoe.stealth,
    toe: tiptoe.circle,
    (0, 0), (calc.cos(10deg), calc.sin(10deg)),
    stroke: (paint: purple, thickness: 0.1em),
    label: [$y$]
  ),
  lq.line(
    (0, 0), (calc.cos(25deg), calc.sin(25deg)),
    stroke: (paint: gray, thickness: 0.1em, dash: "dotted"),
  ),
  lq.line(
    (0, 0), (calc.cos(-25deg), calc.sin(-25deg)),
    stroke: (paint: gray, thickness: 0.1em, dash: "dotted"),
  )


)

== $M$ increases

#let mu1 = 0
#let sigma1 = 0.2
#let mu2 = 2
#let sigma2 = 0.2
#let threshold = 1.2

#let xs = lq.linspace(-5, 8, num: 1000)
#let xfa = lq.linspace(threshold, 8, num: 200)
#let xmiss = lq.linspace(-5, threshold, num: 200)

#lq.diagram(
  grid: none,
  width: 25cm,
  height: 10cm,
  bounds: "strict",
  lq.plot(
    label: [$H_0$],
    xs,
    xs.map(x =>
      0.5*calc.exp(-(x - mu1)*(x - mu1) / (2 * sigma1*sigma1))
    ),
    mark: none,
    stroke: 0.2em
  ),
    lq.plot(
    label: [$H_1$],
    xs,
    xs.map(x =>
      0.5*calc.exp(-(x - mu2)*(x - mu2) / (2 * sigma2*sigma2))
    ),
    mark: none,
    stroke: 0.2em
  ),
)

== $||w||_2^2$ increases

#let mu1 = 0
#let sigma1 = 1
#let mu2 = 5
#let sigma2 = 1
#let threshold = 5

#let xs = lq.linspace(-5, 8, num: 1000)
#let xfa = lq.linspace(threshold, 8, num: 200)
#let xmiss = lq.linspace(-5, threshold, num: 200)

#lq.diagram(
  grid: none,
  width: 25cm,
  height: 10cm,
  bounds: "strict",
  lq.plot(
    label: [$H_0$],
    xs,
    xs.map(x =>
      0.5*calc.exp(-(x - mu1)*(x - mu1) / (2 * sigma1*sigma1))
    ),
    mark: none,
    stroke: 0.2em
  ),
    lq.plot(
    label: [$H_1$],
    xs,
    xs.map(x =>
      0.5*calc.exp(-(x - mu2)*(x - mu2) / (2 * sigma2*sigma2))
    ),
    mark: none,
    stroke: 0.2em
  ),
)

== $sigma_x^2$ increases

#let mu1 = 0
#let sigma1 = 1.5
#let mu2 = 3
#let sigma2 = 1.5
#let threshold = 3

#let xs = lq.linspace(-5, 8, num: 1000)
#let xfa = lq.linspace(threshold, 8, num: 200)
#let xmiss = lq.linspace(-5, threshold, num: 200)

#lq.diagram(
  grid: none,
  width: 25cm,
  height: 10cm,
  bounds: "strict",
  lq.plot(
    label: [$H_0$],
    xs,
    xs.map(x =>
      0.5*calc.exp(-(x - mu1)*(x - mu1) / (2 * sigma1*sigma1))
    ),
    mark: none,
    stroke: 0.2em
  ),
    lq.plot(
    label: [$H_1$],
    xs,
    xs.map(x =>
      0.5*calc.exp(-(x - mu2)*(x - mu2) / (2 * sigma2*sigma2))
    ),
    mark: none,
    stroke: 0.2em
  ),
)

= Multiple bit data hiding

#figure(
  {
  let size = (4, 5)
  let nodes = (
    image-node("input", src: src_dir + "/img/tangled_2.png", cover: true, title: "", image-size: size, pos: explicit-pos(0, y: 0)),

    module("embedder", title: [Embedder], pos: right-of("input", by: 2)),
    module("encoder", title: [Encoder], pos: above("embedder")),
    group("embedding-encoder", ("embedder", "encoder"), title: ""),

    text-node("m", [Message *$m$*], pos: above("encoder")),
    text-node("key_1", [#emoji.key], pos: below("embedder")),

    image-node("embed", src: src_dir + "/img/tangled_2.png", cover: true, title: "", image-size: size, pos: right-of("embedder")),
    module("public", title: [Attacks], pos: right-of("embed")),
    image-node("distorted", src: src_dir + "/img/attacked.png", cover: true, title: "", image-size: size, pos: right-of("public")),

    module("extractor", title: [Extractor], pos: right-of("distorted")),
    module("decoder", title: [Decoder], pos: above("extractor")),
    group("extractor-decoder", ("extractor", "decoder"), title: ""),

    text-node("hat-m", [Message *$hat(m)$*], pos: above("decoder")),
    text-node("key_2", [#emoji.key], pos: below("extractor")),

  )

  let edges = (
    ml-edge("input", "embedder", label: [*$x$*], label-side: left, label-pos: 25%),
    ml-edge("input", "encoder", mark: "-|>", from-side: "right", to-side: "left", via: ((1, 0), (1, -1))),
    ml-edge("embedder", "embed", label: [*$y$*]),
    ml-edge("embed", "public"),
    ml-edge("public", "distorted"),
    ml-edge("distorted", "extractor", label: [*$hat(y)$*]),

    ml-edge("m", "encoder"),
    ml-edge("encoder", "embedder"),

    ml-edge("extractor", "decoder"),
    ml-edge("decoder", "hat-m"),

    ml-edge("key_1", "embedder", label: [*$k$*]),
    ml-edge("key_2", "extractor", label: [*$k$*]),

  )

  ml-diagram(nodes, edges: edges, label-size: 1em, spacing: 1.6em)
  },
)

== Random vs Periodic

#tblock(title: "Random")[
  - Pseudo-random embedding positions determined by a key
  - More secure against targeted removal
  - More vulnerable to geometric attacks
]

#tblock(title: "Periodic")[
  - Embedding at fixed intervals (e.g., every 8 pixels)
  - More robust to geometric attacks (self-synchronization)
  - More vulnerable to targeted removal
]

#tblock(title: "Error correction (Repetition-Based and ECC encoding)")[
  - Increase robustness
]

= Multi-bit spread spectrum watermarking

#figure(
  {
    let size = (5, 1.5)
    let nodes = (
      module("w_1", title: $w_1$, size: size, pos: explicit-pos(0, y: 0)),
      module("w_2", title: $w_2$, size: size, pos: below("w_1")),
      text-node("dots", $dots$, pos: below("w_2"), node-size: size),
      module("w_L", title: $w_L$, size: size, pos: below("dots")),
      group("key", ("w_1", "w_2", "w_L"), title: [Generated by Key], title-pos: "top"),

      text-node("m_1", $m_1 times$, pos: left-of("w_1", by: 0.8)),
      text-node("m_2", $m_2 times$, pos: left-of("w_2", by: 0.8)),
      text-node("m_L", $m_L times$, pos: left-of("w_L", by: 0.8)),
      text-node("m", $m_i in {0, 1}$, pos: below("m_L")),

      gate-node("xor", dy: 6pt, symbol: "add",pos: right-of("w_2", by: 2, dy: 0.5)),

      module("w", title: $w$, size: size, pos: right-of("xor", by: 2)),

      text-node("watermarked", $y = x + alpha w$, pos: right-of("w")),
    )
    let edges = (
      ml-edge("w_1", "xor", orthogonal: true),
      ml-edge("w_2", "xor", via: ((1, 1), (1, 1.5))),
      ml-edge("w_L", "xor", orthogonal: true),
      ml-edge("xor", "w"),

    )
    ml-diagram(nodes, edges: edges, label-size: 1em, spacing: 1.6em)
  }
)

== Detection

- Each carrier is orthogonal to the others:
  - $w_i^T w_j = 0$ for $i != j$
  - $w_i^T w_i = ||w_i||_2^2 > 0$
- Each carrier is detected independently:
  - $S_i = y^T w_i$
  - Taking the sign of $S_i$ gives the bit value $hat(m_i)$
- Link to zero-bit detection:
  - Similar to performing $L$ independent zero-bit detections, one for each carrier

= Perceptual masking

#figure(
  {
    let nodes = (
      image-node("input", src: src_dir + "/img/tangled_2.png", cover: true, title: "", image-size: (5, 6), pos: explicit-pos(0, y: 0)),

      image-node("nvf", src: src_dir + "/img/nvf.png", cover: true, title: "", image-size: (5, 6), pos: right-of("input", by: 2, dy: 0.7)),
      image-node("mask", src: src_dir + "/img/ref_weighted.png", cover: true, title: "", image-size: (5, 6), pos: right-of("nvf")),
      text-node("nvf-label", [Perceptual Masking], pos: below("nvf", by: 0.6)),

      image-node("ref", src: src_dir + "/img/ref.png", cover: true, title: "", image-size: (5, 6), pos: right-of("input", by: 2, dy: -0.7)),
      text-node("ref-label", [No Perceptual Masking], pos: above("ref", by: 0.8)),
    )
    let edges = (
      ml-edge("input", "nvf", via: ((1, 0), (1, 0.7))),
      ml-edge("input", "ref", via: ((1, 0), (1, -0.7))),
      ml-edge("nvf", "mask"),
    )

    ml-diagram(nodes, edges: edges, label-size: 1em, spacing: 1.4em)
  }
)

#pagebreak()

#tblock(title: "Local Variance")[
  $ sigma^2(x) = 1/W^2 sum_(i=1)^W sum_(j=1)^W x_(i j)^2 - (1/W^2 sum_(i=1)^W sum_(j=1)^W x_(i j)) ^2 $
]

#tblock(title: "Noise visibility function")[
  $ text("NVF") = 1/(1 + D dot (sigma^2(x))/sigma^2_max) $
]

#pagebreak()

#tblock(title: "Weighted Mean Squared Error")[
  $ text("WMSE") = 1 / (H dot W) sum_(i=1)^H sum_(j=1)^W text("NVF")_(i j)(x_(i j) - y_(i j))^2 $
]

#tblock(title: "Weighted Peak Signal to Noise Ratio")[
  $ text("WPSNR") = 10 log_10 (text("MAX")^2 / text("WMSE")) $
]

#tblock(title: "Embedding with perceptual masking")[
  $ y = x + alpha_1 text("NVF") dot w + alpha_2 (1 - text("NVF")) dot w text("        with") alpha_1 <  alpha_2 $
]

= Embedding domain

#tblock(title: "Spatial domain")[
  - Directly modify pixel values
  - Fast and simple
]

#tblock(title: "Frequency domain")[
  - Modify coefficients in a transform domain (e.g., DCT, DFT, DWT)
  - Sensitive to affine transformations
  - Good for compression robustness
  - Fast but requires transform computation
]

#pagebreak()

#tblock(title: "Learnable domain")[
  - Linear (e.g., PCA)
    - Energy compaction
    - Linear => limited robustness to complex transformations
  - Non-linear (e.g., Autoencoders)
    - Robust to noise and minor geometric distortions
    - Flexible and adaptable to different requirements
    - But requires training and tuning
  - Foundation models (e.g., CLIP)
    - High robustness to a wide range of transformations
    - Allows to exploit pretrained models without extensive retraining
    - Computationally expensive and vulnerable to adversarial attacks
]

== Autoencoder-based watermarking

#figure(
  {
  let size = (3, 4)
  let nodes = (
    image-node("input", src: src_dir + "/img/tangled_2.png", cover: true, title: $x$, caption-pos: "top", title-gap: 20pt, image-size: size, pos: explicit-pos(0, y: 0)),
    encoder("encoder", title: $f_phi$, pos: right-of("input", by: 2), size: (3, 2)),
    text-node("lock", [#emoji.lock.open], pos: below("encoder", by: 0.5)),
    module("embedder", title: [WM \ Enc.], pos: right-of("encoder")),
    decoder("decoder", title: $g_phi$, pos: right-of("embedder"), size: (3, 2)),
    text-node("lock-2", [#emoji.lock.open], pos: below("decoder", by: 0.5)),
    image-node("watermarked", src: src_dir + "/img/tangled_2.png", cover: true, title: $y$, caption-pos: "top", title-gap: 20pt, image-size: size, pos: right-of("decoder")),
    module("attack", title: $cal(A)$, pos: right-of("watermarked")),
    image-node("distorted", src: src_dir + "/img/attacked.png", cover: true, title: "", image-size: size, pos: right-of("attack")),
    encoder("encoder_2", title: $f_phi$, pos: right-of("distorted"), size: (3, 2)),
    text-node("lock-3", [#emoji.lock.open], pos: below("encoder_2", by: 0.5)),
    module("extractor", title: [WM \ Dec.], pos: right-of("encoder_2")),

    compare-node("compare", title: "", pos: right-of("extractor", dy: -0.1)),
    text-node("compare-label", $cal(L)_m (m, hat(m))$, pos: below("compare", by: 0.5)),

    compare-node("compare_2", title: "", pos: below("attack")),
    text-node("compare_2-label", $cal(L)_x (x, y)$, pos: below("compare_2", by: 0.7)),

    text-node("m", $m$, pos: above("embedder")),



  )

  let edges = (
    ml-edge("input", "encoder"),
    ml-edge("encoder", "embedder"),
    ml-edge("embedder", "decoder"),
    ml-edge("decoder", "watermarked"),
    ml-edge("watermarked", "attack"),
    ml-edge("attack", "distorted"),
    ml-edge("distorted", "encoder_2"),
    ml-edge("encoder_2", "extractor"),
    ml-edge("extractor", "compare", to-shift: -0.1),

    ml-edge("m", "embedder"),
    ml-edge("m", "compare", to-shift: -0.1, via: ((9.4, -1), (9.4, -0.1))),

    ml-edge("input", "compare_2", orthogonal: "v", to-shift: 0.1),
    ml-edge("watermarked", "compare_2", orthogonal: "v", to-shift: -0.1),

  )

  ml-diagram(nodes, edges: edges, label-size: 1em, spacing: 1em)
  },
)

== Foundation model-based watermarking

#figure(
  {
  let size = (3, 4)
  let nodes = (
    image-node("input", src: src_dir + "/img/tangled_2.png", cover: true, title: $x$, caption-pos: "top", title-gap: 20pt, image-size: size, pos: explicit-pos(0, y: 0)),
    encoder("encoder", title: $f_phi$, pos: right-of("input", by: 2), size: (3, 2)),
    text-node("lock", [#emoji.lock], pos: below("encoder", by: 0.5)),
    module("embedder", title: [WM \ Enc.], pos: right-of("encoder")),
    decoder("decoder", title: $g_phi$, pos: right-of("embedder"), size: (3, 2)),
    text-node("lock-2", [#emoji.lock.open], pos: below("decoder", by: 0.5)),
    image-node("watermarked", src: src_dir + "/img/tangled_2.png", cover: true, title: $y$, caption-pos: "top", title-gap: 20pt, image-size: size, pos: right-of("decoder")),
    module("attack", title: $cal(A)$, pos: right-of("watermarked")),
    image-node("distorted", src: src_dir + "/img/attacked.png", cover: true, title: "", image-size: size, pos: right-of("attack")),
    encoder("encoder_2", title: $f_phi$, pos: right-of("distorted"), size: (3, 2)),
    text-node("lock-3", [#emoji.lock], pos: below("encoder_2", by: 0.5)),
    module("extractor", title: [WM \ Dec.], pos: right-of("encoder_2")),

    compare-node("compare", title: "", pos: right-of("extractor", dy: -0.1)),
    text-node("compare-label", $cal(L)_m (m, hat(m))$, pos: below("compare", by: 0.5)),

    compare-node("compare_2", title: "", pos: below("attack")),
    text-node("compare_2-label", $cal(L)_x (x, y)$, pos: below("compare_2", by: 0.7)),

    text-node("m", $m$, pos: above("embedder")),



  )

  let edges = (
    ml-edge("input", "encoder"),
    ml-edge("encoder", "embedder"),
    ml-edge("embedder", "decoder"),
    ml-edge("decoder", "watermarked"),
    ml-edge("watermarked", "attack"),
    ml-edge("attack", "distorted"),
    ml-edge("distorted", "encoder_2"),
    ml-edge("encoder_2", "extractor"),
    ml-edge("extractor", "compare", to-shift: -0.1),

    ml-edge("m", "embedder"),
    ml-edge("m", "compare", to-shift: -0.1, via: ((9.4, -1), (9.4, -0.1))),

    ml-edge("input", "compare_2", orthogonal: "v", to-shift: 0.1),
    ml-edge("watermarked", "compare_2", orthogonal: "v", to-shift: -0.1),

  )

  ml-diagram(nodes, edges: edges, label-size: 1em, spacing: 1em)
  },
)

= Geometrical synchronization

#tblock(title: "Hand-crafted invariant domain watermarking")[
  - Embed in a domain invariant to geometric transformations
    - Fourier (translation), Fourier-Mellin (rotation, scale, translation)
]

#tblock(title: "Feature-based synchronization")[
  - Use robust features (e.g., SIFT) (requires exact prior knowledge)
]

#tblock(title: "Self-synchronizing watermarking")[
  - Repeated structure (security leak, size constraints, perceptual visibility)
]

#tblock(title: "Foundation model-based synchronization")[
  - Use pretrained invariant models (e.g., CLIP, DINO) (adversarial + computational cost)
]

= Additive vs Quantization approaches

#tblock(title: "Additive")[
  - Simple and flexible
  - Sensitive to host interference
]

#tblock(title: "Binary Quantization Index Modulation")[
  - More robust to noise
  - Sensitive to scaling and volumetric operations
]

#pagebreak()

#tblock(title: "Vector Quantization")[
  - Extension of QIM to multiple bits
  - Better performance but more complex
]

#tblock(title: "Product Quantization")[
  - Balances robustness and complexity
  - Suitable for large-scale systems
]

= Key management

- Symmetric: Same key for embedding and extraction
- Key reuse: adversary can learn/estimate the key
- Kind of key:
  - positional key: determines embedding positions
  - encoding key: determines the codebook or carriers
  - XOR key: added to the message before encoding
- Problem: which key to use ?
  - Fingerprinting: category of input determines the key
  - Key-message search: find the key that decodes a meaningful message

#figure(
  {
  let size = (3, 4)
  let nodes = (
    image-node("input", src: src_dir + "/img/tangled_2.png", cover: true, title: $x$, caption-pos: "top", title-gap: 20pt, image-size: size, pos: explicit-pos(0, y: 0)),
    encoder("encoder", title: $f_phi$, pos: right-of("input", by: 2), size: (3, 2)),
    text-node("lock", [#emoji.lock], pos: below("encoder", by: 0.5)),
    module("encoding", title: [Encoding], pos: right-of("encoder", by: 2, dy: -1)),
    module("db", title: [DB], pos: right-of("encoding"), shape: shapes.cylinder, size: (3, 1.5)),
    module("embedder", title: [WM \ Encoding], pos: right-of("encoder", by: 2, dy: 1)),
    decoder("decoder", title: $g_phi$, pos: right-of("embedder"), size: (3, 2)),
    text-node("lock-2", [#emoji.lock.open], pos: below("decoder", by: 0.5)),
    module("attack", title: $cal(A)$, pos: explicit-pos(8, y: 0)),
    image-node("distorted", src: src_dir + "/img/attacked.png", cover: true, title: "", image-size: size, pos: right-of("attack")),
    encoder("encoder_2", title: $f_phi$, pos: right-of("distorted"), size: (3, 2)),
    text-node("lock-3", [#emoji.lock], pos: below("encoder_2", by: 0.5)),
    module("extractor", title: [WM \ Dec.], pos: right-of("encoder_2", by: 2, dy: 1)),
    module("key-search", title: [ANN \ Search], pos: right-of("encoder_2", by: 2,dy: -1)),
    text-node("m", $m$, pos: right-of("extractor", by: 2)),
    text-node("hat-m", $hat(m)$, pos: above("m", by: 0.5)),
    text-node("equal", [=?], pos: above("m", by: 0.25)),
    decoder("decoder_2", title: "DST", pos: below("extractor", by: 1.5, dx: 0.2), size: (3, 1.5)),
    text-node("c-hat", $hat(c)$, pos: right-of("decoder_2", by: 1.8)),
    text-node("lock-4", [#emoji.lock], pos: below("decoder_2", by: 1.1)),

    text-node("imk", $[i, m, k]$, pos: above("input", by: 1.8)),
    text-node("i", $i$, pos: right-of("key-search", by: 2, dy: -0.3)),


    group("encoding-db", ("encoding", "db"), title: "Fingerprinting", title-pos: "top", title-shift: (1, -2.3), title-gap: 25pt),
    group("embedding-decoding", ("embedder", "decoder"), title: "Watermarking", title-pos: "bottom", title-shift: (1, 4), title-gap: 25pt),



  )

  let edges = (
    ml-edge("input", "encoder"),
    ml-edge("encoder", "embedder", via: ((3, 0), (3, 1))),
    ml-edge("encoder", "encoding", via: ((3, 0), (3, -1))),
    ml-edge("encoding", "db"),
    ml-edge("embedder", "decoder"),
    ml-edge("decoder", "attack", via: ((7, 1), (7, 0))),
    ml-edge("attack", "distorted"),
    ml-edge("distorted", "encoder_2"),
    ml-edge("encoder_2", "extractor", via: ((11, 0), (11, 1))),
    ml-edge("encoder_2", "key-search", to-shift: 0.2, via: ((11, 0), (11, -1))),
    ml-edge("db", "embedder", to-side: "top", via: ((5, 0), (4, 0) )),
    ml-edge("db", "key-search"),
    ml-edge("key-search", "extractor", to-side: "top", label: $k$, label-side: left),
    ml-edge("extractor", "m"),
    ml-edge("key-search", "hat-m", orthogonal: true),
    ml-edge("encoder_2", "decoder_2", via: ((11, 0), (11, 2.5))),
    ml-edge("decoder_2", "c-hat"),
    ml-edge("imk", "db", orthogonal: true),
    ml-edge("key-search", "i", from-shift: 0.25)



  )

  ml-diagram(nodes, edges: edges, label-size: 1em, spacing: 1em)
  },
)

= Transform domain watermarking

#tblock(title: "Discrete Cosine Transform (DCT)")[
  - Commonly used in JPEG compression
  - Good for robustness against compression
  - Sensitive to geometric transformations
]
#tblock(title: "Discrete Wavelet Transform (DWT)")[
  - Multi-resolution analysis
  - Good for robustness against noise and minor geometric distortions
  - Computationally more complex than DCT
]

#pagebreak()

#tblock(title: "Discrete Fourier Transform (DFT)")[
  - $F(u, v) = sum_(x=1)^M sum_(y=1)^N f(x, y) e^(-j 2 pi((u x)/M + (v y)/N))$
  - Invariant to translation
  - Sensitive to rotation and scaling
  - Computationally expensive
]
#tblock(title: "Fourier-Mellin Transform")[
  - Log-polar mapping of the Fourier transform (degrades quality)
  - $(u, v) -> (log sqrt(u^2 + v^2), tan^(-1)(v/u))$ + 1D Fourier in angular direction
  - Invariant to rotation, scaling, and translation
  - Good for robustness against geometric transformations
  - Computationally expensive and less common in practice
]

= Attacks

#tblock(title: "Removal attacks")[
  - Denoising and noise addition
  - Averaging
  - Collusion (estimate watermark from each image -> align -> suppress)
]

#tblock(title: "Copy attacks")[
  - Copy the watermark from one to another image
]

== Adversarial copy attack

#figure(
  {
    let size = (3, 2)
    let embedding-size = (1.5, 3)

    let nodes = (
      text-node("x_w", $x_w$, pos: explicit-pos(0, y: 0)),
      encoder("encoder", title: $f_phi$, pos: right-of("x_w", by: 2), size: size),
      text-node("lock", [#emoji.lock], pos: below("encoder", by: 0.8)),
      text-node("z_w", $z_w$, pos: right-of("encoder"), node-size: (2, 2)),
      decoder("decoder", title: $g_phi$, pos: right-of("z_w"), size: size),
      text-node("lock-2", [#emoji.lock.open], pos: below("decoder", by: 0.8)),
      text-node("x_a", $x_a$, pos: right-of("decoder"), node-size: (2, 2)),
      compare-node("compare_x", title: "", pos: right-of("x_a", dy: -1)),
      text-node("compare_x-label", $cal(L)_x (x_a, x_w)$, pos: below("compare_x", by: 0.5, dx: 0.5)),
      text-node("x_t", $x_t$, pos: above("x_w", by: 1.2)),

      text-node("x_a_2", $x_a$, pos: below("x_w", by: 2), node-size: (2, 2)),
      encoder("encoder_2", title: $f_phi$, pos: right-of("x_a_2", by: 2), size: size),
      text-node("lock-3", [#emoji.lock], pos: below("encoder_2", by: 0.9)),
      text-node("z_a", $z_a$, pos: right-of("encoder_2"), node-size: (2, 2)),

      compare-node("compare_z", title: "", pos: right-of("z_a", dy: -0.3)),
      text-node("compare_z-label", $cal(L)_z (z_a, z_w)$, pos: below("compare_z", by: 0.5, dx: 0.4))


    )

    let edges = (
      ml-edge("x_w", "encoder"),
      ml-edge("encoder", "z_w", mark: "-"),
      ml-edge("z_w", "decoder"),
      ml-edge("decoder", "x_a"),


      ml-edge("x_a_2", "encoder_2"),
      ml-edge("encoder_2", "z_a", mark: "-"),

      ml-edge("x_t", "compare_x", to-shift: 0.2),
      ml-edge("x_a", "compare_x", orthogonal: "v", to-shift: 0.2),

      ml-edge("z_a", "compare_z", to-shift: -0.3),
      ml-edge("z_w", "compare_z", orthogonal: "v", to-shift: -0.2),
      ml-edge("x_a_2", "x_a", via: ((0, 5), (5, 5)), mark: "-"),
  
    )

    ml-diagram(nodes, edges: edges, label-size: 1em, spacing: 1.3em)
  }
)

== Adversarial removal attack

#figure(
  {
    let size = (3, 2)
    let embedding-size = (1.5, 3)

    let nodes = (
      text-node("x_w", $x_w$, pos: explicit-pos(0, y: 0), node-size: (2, 2)),
      encoder("encoder", title: $f_phi$, pos: right-of("x_w", by: 2), size: size),
      text-node("lock", [#emoji.lock], pos: below("encoder", by: 0.8)),
      text-node("z_w", $z_w$, pos: right-of("encoder"), node-size: (2, 2)),
      decoder("decoder", title: $g_phi$, pos: right-of("z_w"), size: size),
      text-node("lock-2", [#emoji.lock.open], pos: below("decoder", by: 0.8)),
      text-node("x_a", $x_a$, pos: right-of("decoder"), node-size: (2, 2)),
      compare-node("compare_x", title: "", pos: right-of("x_a", dy: -0.6)),
      text-node("compare_x-label", $cal(L)_x (x_a, x_w)$, pos: below("compare_x", by: 0.5, dx: 0.5)),

      text-node("x_a_2", $x_a$, pos: below("x_w", by: 4), node-size: (2, 2)),
      encoder("encoder_2", title: $f_phi$, pos: right-of("x_a_2", by: 2), size: size),
      text-node("lock-3", [#emoji.lock], pos: below("encoder_2", by: 0.9)),
      text-node("z_a", $z_a$, pos: right-of("encoder_2"), node-size: (2, 2)),

      compare-node("compare_z", title: "", pos: right-of("z_a", dy: -0.3)),
      text-node("compare_z-label", $cal(L)_z (z_a, z_w)$, pos: below("compare_z", by: 0.5, dx: 0.4)),

      module("transform", title: $T$, pos: below("x_w")),
      text-node("x_t", $x_t$, pos: below("transform")),
      encoder("encoder_t", title: $f_phi$, pos: right-of("x_t", by: 2), size: size),
      text-node("lock-4", [#emoji.lock], pos: below("encoder_t", by: 0.8)),
      text-node("z_t", $z_t$, pos: right-of("encoder_t"), node-size: (2, 2)),

      text-node("ghost", "", pos: right-of("x_t", by: -1.5))


    )

    let edges = (
      ml-edge("x_w", "encoder"),
      ml-edge("encoder", "z_w", mark: "-"),
      ml-edge("z_w", "decoder"),
      ml-edge("decoder", "x_a"),


      ml-edge("x_a_2", "encoder_2"),
      ml-edge("encoder_2", "z_a", mark: "-"),
      ml-edge("x_w", "compare_x", orthogonal: "v", to-shift: -0.2),

      ml-edge("x_a", "compare_x", orthogonal: "v", to-shift: 0.2),

      ml-edge("z_a", "compare_z", to-shift: -0.3),
      ml-edge("z_t", "compare_z", orthogonal: "v", to-shift: -0.2),
      ml-edge("x_a_2", "x_a", via: ((0, 5.3), (5, 5.3)), mark: "-"),

      ml-edge("x_w", "transform"),
      ml-edge("transform", "x_t"),
      ml-edge("x_t", "encoder_t"),

      ml-edge("encoder_t", "z_t", mark: "-"),

      ml-edge("ghost", "x_t", dash: "dashed")
  
    )

    ml-diagram(nodes, edges: edges, label-size: 1em, spacing: 1.3em)
  }
)

= Perceptual vs cryptographic hashing

#tblock(title: "Cryptographic hashing")[
  - Designed to be collision-resistant and unpredictable
  - Small changes in input lead to large changes in output (avalanche effect)
  - Not suitable for perceptual similarity tasks
]

#tblock(title: "Perceptual hashing")[
  - Designed to produce similar outputs for perceptually similar inputs
  - Robust to minor modifications (e.g., resizing, compression)
  - Used for tasks like image retrieval and copyright enforcement
  - Hypothesis testing:
    - $H_0$: $y = x + epsilon$
    - $H_1$: $y = x(m) + epsilon$
]

#let mu1 = 0
#let sigma1 = 1
#let mu2 = 5
#let mu_informed = 2
#let sigma2 = 1
#let threshold = 5

#let xs = lq.linspace(-5, 8, num: 1000)
#let xfa = lq.linspace(threshold, 8, num: 200)
#let xmiss = lq.linspace(-5, threshold, num: 200)

#lq.diagram(
  grid: none,
  width: 25cm,
  height: 10cm,
  bounds: "strict",
  lq.plot(
    label: [$H_0$],
    xs,
    xs.map(x =>
      0.2*calc.exp(-(x - mu1)*(x - mu1) / (2 * sigma1*sigma1))
    ),
    mark: none,
    stroke: 0.2em
  ),
    lq.plot(
    label: [$H_1$],
    xs,
    xs.map(x =>
      0.4*calc.exp(-(x - mu2)*(x - mu2) / (2 * sigma2*sigma2))
    ),
    mark: none,
    stroke: 0.2em
  ),
  lq.plot(
    label: [$H_text("informed")$],
    xs,
    xs.map(x =>
      0.3*calc.exp(-(x - mu_informed)*(x - mu_informed) / (2 * sigma2*sigma2))
    ),
    mark: none,
    stroke: 0.2em
  ),
)

= Content fingerprinting

#tblock(title: "Definition")[
  - Extract a unique identifier (fingerprint) from the content
  - Used for tracking and identifying content without modifying it
  - Challenges:
    - Robustness to transformations and attacks
    - Uniqueness and collision resistance
    - Privacy concerns
]

#tblock(title: "Applications")[
  - Copyright
  - Image and video identification
  - Content-based retrieval
]

= Training of Foundation models

#tblock(title: "Embedding Reconstruction (AE)")[
  - No mode collapse
  - High complexity
  - No good loss for pixel space
]
#tblock(title: "Embedding Reconstruction (CLIP)")[
  - Mode collapse (many images can have the same embedding)
  - Low complexity
  - Good loss for latent space
]

== Hybrid approaches

#figure(
  {
    let nodes = (
      image-node("input", src: src_dir + "/img/tangled_2.png", cover: true, title: $x_0$, caption-pos: "top", title-gap: 20pt, image-size: (3, 4), pos: explicit-pos(0, y: 0)),
      module("transform", title: $T$, pos: right-of("input")),
      image-node("flipped", src: src_dir + "/img/tangled_2_flipped.png", cover: true, title: "", image-size: (3, 4), pos: right-of("transform")),
      text-node("flipped-label", $x$, pos: above("flipped", by: 0.7, dx: -0.2)),
      module("mask", title: $M$, pos: below("flipped")),
      text-node("m_a", $m_a$, pos: left-of("mask")),
      image-node("masked", src: src_dir + "/img/tangled_2_masked.png", cover: true, title: $x_a$, caption-pos: "bottom", title-gap: 15pt, image-size: (3, 4), pos: below("mask")),
      encoder("encoder", title: $f_psi$, pos: right-of("masked"), size: (3, 2)),
      text-node("lock", [#emoji.lock.open], pos: below("encoder", by: 0.6)),
      decoder("decoder", title: $g_phi$, pos: right-of("encoder", by: 2), size: (3, 2)),
      text-node("lock-2", [#emoji.lock.open], pos: below("decoder", by: 0.6)),
      image-node("reconstructed", src: src_dir + "/img/tangled_2_flipped.png", cover: true, title: $hat(x)_a$, caption-pos: "bottom", title-gap: 15pt, pos: right-of("decoder"), image-size: (3, 4)),

      compare-node("compare", title: "", pos: right-of("reconstructed", by: 1, dy: -1)),
      text-node("compare-label", $cal(L)_x (hat(x)_a, x_a)$, pos: below("compare", by: 0.5)),

      encoder("encoder_2", title: $f_phi$, pos: right-of("flipped"), size: (3, 2)),
      text-node("lock-3", [#emoji.lock], pos: below("encoder_2", by: 0.4)),

      compare-node("compare_2", title: "", pos: right-of("encoder_2", by: 1, dy: 1)),
      text-node("compare_2-label", $cal(L)_z (y_x, y_x_a)$, pos: below("compare_2", by: 0.3, dx: 0.5)),

    )

    let edges = (
      ml-edge("input", "transform"),
      ml-edge("transform", "flipped"),
      ml-edge("flipped", "mask"),
      ml-edge("m_a", "mask"),
      ml-edge("mask", "masked"),
      ml-edge("masked", "encoder"),
      ml-edge("encoder", "decoder"),
      ml-edge("decoder", "reconstructed"),
      ml-edge("reconstructed", "compare", orthogonal: "v", to-shift: 0.15),
      ml-edge("flipped", "compare", via: ( (2, -1), (5, -1), (5, 1)), to-shift: -0.15),

      ml-edge("flipped", "encoder_2"),

      ml-edge("encoder_2", "compare_2", via: ( (3.5, 0), (3.5, 1)), to-shift: -0.15, label: $y_x$, label-side: right, label-pos: 60%),
      ml-edge("encoder", "compare_2", via: ( (3.5, 2), (3.5, 1)), to-shift: 0.15, label: $y_x_a$, label-side: left, label-pos: 60%),


    )

    ml-diagram(nodes, edges: edges, label-size: 1em, spacing: 1.6em)
  }
)

= Main requirements for privacy protection

#tblock(title: "Traditional privacy protection")[
  - Data owner (protect data) (noise addition, data removal, dim reduction)
  - Data user (protect requests) (anonymization, randomized rules)
]

#pagebreak()

== Fuzzy Commitment

#figure(
  {
    let nodes = (
      text-node("key", [#emoji.key], pos: explicit-pos(0, y: 0), node-size: (2, 2)),
      module("ecc", title: $text("ECC")$, pos: right-of("key")),
      gate-node("gap", pos: right-of("ecc")),
      text-node("mxk", $m = k xor b_x$, pos: right-of("gap")),
      gate-node("gap_2", pos: right-of("mxk")),
      module("ecc-1", title: $text("ECC")^(-1)$, pos: right-of("gap_2")),
      module("hash", title: "Hash", pos: right-of("ecc-1")),
      module("equal", title: "Equal ?", pos: right-of("hash", dy: -1)),

      module("fingerprint_x", title: "Fingerprint", pos: below("gap")),
      module("fingerprint_y", title: "Fingerprint", pos: below("gap_2")),
      text-node("x", $x$, pos: below("fingerprint_x")),
      text-node("y", $y$, pos: below("fingerprint_y")),

      module("hash_2", title: "Hash", pos: above("gap", by: 2)),
      text-node("phi", $phi(k)$, pos: right-of("hash_2")),

      group("public", ("phi", "mxk"), title: [Public \ Storage], title-pos: "center", title-shift: (1, 2), title-gap: 25pt),


    )
    let edges = (
      ml-edge("key", "ecc"),
      ml-edge("ecc", "gap", label: $k$, label-side: left),
      ml-edge("gap", "mxk"),
      ml-edge("mxk", "gap_2"),
      ml-edge("gap_2", "ecc-1"),
      ml-edge("ecc-1", "hash", label: $hat(k)$),
      ml-edge("hash", "equal", orthogonal: true, label: $phi(hat(k))$, label-pos: 20%, label-sep: 2pt),

      ml-edge("x", "fingerprint_x"),
      ml-edge("y", "fingerprint_y"),
      ml-edge("fingerprint_x", "gap", label: $b_x$, label-side: left),
      ml-edge("fingerprint_y", "gap_2", label: $b_y$, label-side: right),

      ml-edge("key", "hash_2", orthogonal: "v"),
      ml-edge("hash_2", "phi"),
      ml-edge("phi", "equal", orthogonal: true),

    )

    ml-diagram(nodes, edges: edges, label-size: 1em, spacing: 1.6em)
  }
)

== Helper data

#figure(
  {
    let nodes = (
      text-node("x", $x$, pos: explicit-pos(0, y: 0)),
      module("fingerprint_x", title: "Fingerprint", pos: right-of("x")),
      text-node("b_x", $b_x$, pos: right-of("fingerprint_x")),
      module("hash_x", title: "Hash", pos: right-of("b_x", by: 3)),
      module("ecc", title: $text("ECC")$, pos: right-of("b_x", dy: 0.6)),
      
      text-node("y", $y$, pos: below("x", by: 2)),
      module("fingerprint_y", title: "Fingerprint", pos: right-of("y")),
      text-node("b_y", $b_y$, pos: right-of("fingerprint_y")),
      module("ecc-1", title: $text("ECC")^(-1)$, pos: right-of("b_y")),
      text-node("b_x_hat", $hat(b)_x$, pos: right-of("ecc-1")),
      module("hash_y", title: "Hash", pos: right-of("b_x_hat")),

      module("equal", title: "Equal ?", pos: right-of("hash_x", by: 2, dy: 1)),

    )
    let edges = (
      ml-edge("x", "fingerprint_x"),
      ml-edge("b_x", "ecc", orthogonal: true),
      ml-edge("fingerprint_x", "b_x", mark: "-"),
      ml-edge("b_x", "hash_x"),

      ml-edge("y", "fingerprint_y"),
      ml-edge("fingerprint_y", "b_y", mark: "-"),
      ml-edge("b_y", "ecc-1"),
      ml-edge("ecc-1", "b_x_hat", mark: "-"),
      ml-edge("b_x_hat", "hash_y"),

      ml-edge("hash_x", "equal", orthogonal: true, label: $phi(k)$, label-pos: 20%, label-sep: 2pt),
      ml-edge("hash_y", "equal", orthogonal: true, label: $phi(hat(k))$, label-pos: 20%, label-sep: 8pt, label-side: right),
      ml-edge("ecc", "ecc-1", label: [Helper data], label-side: center)
    )

    ml-diagram(nodes, edges: edges, label-size: 1em, spacing: 1.6em)
  }
)

Redundancy is proportional to $L H_2 (P_b)$ !
