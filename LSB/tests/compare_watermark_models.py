from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import inspect
import itertools
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[1]
TESTS_DIR = Path(__file__).resolve().parent


@dataclass(frozen=True)
class ModelSpec:
    name: str
    path: Path
    ecc_modes: tuple[str, ...]


MODEL_SPECS: tuple[ModelSpec, ...] = (
    ModelSpec("codex", ROOT_DIR / "codex" / "model.py", ("hamming", "crc", "crc_hamming")),
    ModelSpec("codex-4", ROOT_DIR / "codex-4" / "model.py", ("single", "combined")),
    ModelSpec("sonnet", ROOT_DIR / "sonnet" / "model.py", ("single", "combined")),
)

ATTACK_NAMES = (
    "none",
    "rotation",
    "grayscale",
    "contrast",
    "brightness",
    "hue",
    "hflip",
    "vflip",
    "blur",
    "jpeg",
)


def parse_int_list(value: str) -> list[int]:
    out: list[int] = []
    for token in value.split(","):
        token = token.strip()
        if not token:
            continue
        out.append(int(token))
    if not out:
        raise ValueError("Expected at least one integer value.")
    return out


def parse_float_list(value: str) -> list[float]:
    out: list[float] = []
    for token in value.split(","):
        token = token.strip()
        if not token:
            continue
        out.append(float(token))
    if not out:
        raise ValueError("Expected at least one float value.")
    return out


def parse_str_list(value: str) -> list[str]:
    out = [token.strip() for token in value.split(",") if token.strip()]
    if not out:
        raise ValueError("Expected at least one string value.")
    return out


def parse_key(value: str) -> int | str:
    stripped = value.strip()
    if stripped and (stripped.isdigit() or (stripped.startswith("-") and stripped[1:].isdigit())):
        return int(stripped)
    return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare LSB watermark model.py implementations (codex, codex-4, sonnet) "
            "with BER/PSNR/runtime metrics and parameter sweeps."
        )
    )
    parser.add_argument(
        "--models",
        default=",".join(spec.name for spec in MODEL_SPECS),
        help="Comma-separated implementation names to benchmark (codex,codex-4,sonnet).",
    )
    parser.add_argument("--images-dir", default=str(ROOT_DIR / "img"), help="Directory containing test images.")
    parser.add_argument("--out-dir", default="results", help="Output directory for CSV/JSON reports.")
    parser.add_argument("--message-lengths", default="32,64,128", help="Comma-separated message lengths.")
    parser.add_argument("--repeats", default="1,3,5", help="Comma-separated repeat values.")
    parser.add_argument("--alphas", default="2.5,4.0,6.0", help="Comma-separated alpha values.")
    parser.add_argument("--local-statistics", default="mean,median", help="codex local_statistic values.")
    parser.add_argument("--neighborhood-sizes", default="3,5", help="sonnet neighborhood_size values.")
    parser.add_argument("--rotation-degrees", type=float, default=10.0, help="Rotation attack angle in degrees.")
    parser.add_argument("--contrast-factor", type=float, default=1.35, help="Contrast attack factor.")
    parser.add_argument("--brightness-shift", type=float, default=25.0, help="Brightness shift.")
    parser.add_argument("--hue-shift", type=int, default=15, help="Hue shift in HSV space [0, 179].")
    parser.add_argument("--blur-kernel", type=int, default=5, help="Odd kernel size for Gaussian blur.")
    parser.add_argument("--jpeg-quality", type=int, default=50, help="JPEG quality for compression attack (1-100).")
    parser.add_argument("--psnr-threshold", type=float, default=30.0, help="PSNR threshold for BER penalty.")
    parser.add_argument("--max-encode-time", type=float, default=5.0, help="Passed to model constructors if supported.")
    parser.add_argument("--max-decode-time", type=float, default=1.0, help="Passed to model constructors if supported.")
    parser.add_argument("--key", default="1337", help="Shared key for all models.")
    parser.add_argument("--seed", type=int, default=12345, help="Global seed for deterministic watermark generation.")
    parser.add_argument("--top-k", type=int, default=5, help="Number of best configs to print per model.")
    parser.add_argument(
        "--max-configs-per-model",
        type=int,
        default=0,
        help="If > 0, truncate generated parameter combinations per model.",
    )
    parser.add_argument("--quiet", action="store_true", help="Reduce progress logging.")
    return parser.parse_args()


def resolve_model_specs(selected_names: list[str]) -> list[ModelSpec]:
    """Resolve selected implementation names to model specifications.

    Parameters
    ----------
    selected_names : list[str]
        Implementation names requested on the command line.

    Returns
    -------
    list[ModelSpec]
        Ordered list of model specs matching ``selected_names``.

    Raises
    ------
    ValueError
        If any requested name is unknown.
    """
    by_name = {spec.name: spec for spec in MODEL_SPECS}
    unknown = [name for name in selected_names if name not in by_name]
    if unknown:
        known = ", ".join(sorted(by_name))
        raise ValueError(f"Unknown model(s): {', '.join(unknown)}. Known models: {known}.")
    return [by_name[name] for name in selected_names]


def to_uint8_image(image: np.ndarray) -> np.ndarray:
    arr = np.asarray(image)
    if arr.dtype == np.uint8:
        return arr
    return np.clip(np.rint(arr), 0, 255).astype(np.uint8)


def ensure_color_image(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if image.ndim == 3 and image.shape[2] == 4:
        return image[:, :, :3]
    if image.ndim == 3 and image.shape[2] == 3:
        return image
    raise ValueError(f"Unsupported image shape: {image.shape}")


def load_images(images_dir: Path) -> list[tuple[str, np.ndarray]]:
    if not images_dir.exists():
        raise FileNotFoundError(f"Image directory not found: {images_dir}")

    patterns = ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.webp", "*.tif", "*.tiff")
    image_paths: list[Path] = []
    for pattern in patterns:
        image_paths.extend(images_dir.glob(pattern))

    unique_paths = sorted(set(image_paths))
    if not unique_paths:
        raise RuntimeError(f"No images found in {images_dir}")

    loaded: list[tuple[str, np.ndarray]] = []
    for image_path in unique_paths:
        arr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if arr is None:
            continue
        loaded.append((image_path.name, to_uint8_image(ensure_color_image(arr))))

    if not loaded:
        raise RuntimeError(f"Could not load any image from {images_dir}")
    return loaded


def psnr_db(reference: np.ndarray, compared: np.ndarray) -> float:
    ref = reference.astype(np.float32, copy=False)
    cmp = compared.astype(np.float32, copy=False)
    mse = float(np.mean((ref - cmp) ** 2, dtype=np.float64))
    if mse <= 0.0:
        return float("inf")
    return float(10.0 * np.log10((255.0 * 255.0) / mse))


def stable_seed(*tokens: Any) -> int:
    joined = "|".join(str(token) for token in tokens)
    digest = hashlib.blake2b(joined.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "little", signed=False)


def deterministic_watermark(image_name: str, message_length: int, global_seed: int) -> np.ndarray:
    rng = np.random.default_rng(stable_seed(global_seed, image_name, message_length))
    return rng.integers(0, 2, size=message_length, dtype=np.uint8)


def ensure_same_layout(attacked: np.ndarray, reference: np.ndarray) -> np.ndarray:
    out = attacked
    if out.shape[:2] != reference.shape[:2]:
        out = cv2.resize(out, (reference.shape[1], reference.shape[0]), interpolation=cv2.INTER_LINEAR)

    if out.ndim == 2:
        out = cv2.cvtColor(out, cv2.COLOR_GRAY2BGR)

    if out.ndim != 3:
        raise ValueError(f"Unsupported attacked image shape: {out.shape}")

    if out.shape[2] > 3:
        out = out[:, :, :3]
    elif out.shape[2] < 3:
        out = np.repeat(out[:, :, :1], 3, axis=2)

    return to_uint8_image(out)


def attack_none(image: np.ndarray, _: argparse.Namespace) -> np.ndarray:
    return image.copy()


def attack_rotation(image: np.ndarray, args: argparse.Namespace) -> np.ndarray:
    h, w = image.shape[:2]
    matrix = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), float(args.rotation_degrees), 1.0)
    return cv2.warpAffine(
        image,
        matrix,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101,
    )


def attack_grayscale(image: np.ndarray, _: argparse.Namespace) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def attack_contrast(image: np.ndarray, args: argparse.Namespace) -> np.ndarray:
    return cv2.convertScaleAbs(image, alpha=float(args.contrast_factor), beta=0)


def attack_brightness(image: np.ndarray, args: argparse.Namespace) -> np.ndarray:
    return cv2.convertScaleAbs(image, alpha=1.0, beta=float(args.brightness_shift))


def attack_hue(image: np.ndarray, args: argparse.Namespace) -> np.ndarray:
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hue_shift = int(args.hue_shift) % 180
    hue = hsv[:, :, 0].astype(np.int16)
    hsv[:, :, 0] = ((hue + hue_shift) % 180).astype(np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def attack_hflip(image: np.ndarray, _: argparse.Namespace) -> np.ndarray:
    return cv2.flip(image, 1)


def attack_vflip(image: np.ndarray, _: argparse.Namespace) -> np.ndarray:
    return cv2.flip(image, 0)


def attack_blur(image: np.ndarray, args: argparse.Namespace) -> np.ndarray:
    kernel = max(1, int(args.blur_kernel))
    if kernel % 2 == 0:
        kernel += 1
    return cv2.GaussianBlur(image, (kernel, kernel), sigmaX=0.0)


def attack_jpeg(image: np.ndarray, args: argparse.Namespace) -> np.ndarray:
    quality = int(np.clip(args.jpeg_quality, 1, 100))
    ok, encoded = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        return image.copy()
    decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if decoded is None:
        return image.copy()
    return decoded


def build_attacks(args: argparse.Namespace) -> dict[str, Callable[[np.ndarray, argparse.Namespace], np.ndarray]]:
    return {
        "none": attack_none,
        "rotation": attack_rotation,
        "grayscale": attack_grayscale,
        "contrast": attack_contrast,
        "brightness": attack_brightness,
        "hue": attack_hue,
        "hflip": attack_hflip,
        "vflip": attack_vflip,
        "blur": attack_blur,
        "jpeg": attack_jpeg,
    }


def normalize_bits(bits: np.ndarray, length: int) -> np.ndarray:
    arr = np.asarray(bits).reshape(-1)
    out = np.zeros(length, dtype=np.uint8)
    if arr.size:
        clipped = arr[:length]
        out[: clipped.size] = (clipped > 0).astype(np.uint8, copy=False)
    return out


def symmetric_ber(reference_bits: np.ndarray, decoded_bits: np.ndarray) -> float:
    if reference_bits.size == 0:
        return 0.0
    raw = float(np.mean(reference_bits != decoded_bits))
    return min(raw, 1.0 - raw)


def load_watermark_class(model_spec: ModelSpec) -> type:
    if not model_spec.path.exists():
        raise FileNotFoundError(f"Missing model file: {model_spec.path}")

    module_name = f"benchmark_{model_spec.name}_{stable_seed(model_spec.path)}"
    module_spec = importlib.util.spec_from_file_location(module_name, model_spec.path)
    if module_spec is None or module_spec.loader is None:
        raise ImportError(f"Could not load module from {model_spec.path}")

    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)

    if not hasattr(module, "WatermarkModel"):
        raise AttributeError(f"{model_spec.path} does not define WatermarkModel")

    klass = getattr(module, "WatermarkModel")
    if not inspect.isclass(klass):
        raise TypeError(f"WatermarkModel in {model_spec.path} is not a class")

    for method_name in ("encode", "decode"):
        method = getattr(klass, method_name, None)
        if method is None or not callable(method):
            raise TypeError(f"WatermarkModel in {model_spec.path} misses callable {method_name}().")

    return klass


def supported_init_params(model_class: type) -> set[str]:
    signature = inspect.signature(model_class.__init__)
    supported: set[str] = set()
    for param_name, param in signature.parameters.items():
        if param_name == "self":
            continue
        if param.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY):
            supported.add(param_name)
    return supported


def compact_config(config: dict[str, Any]) -> str:
    ordered_keys = (
        "message_length",
        "repeat",
        "alpha",
        "use_error_correction",
        "ecc_mode",
        "local_statistic",
        "neighborhood_size",
        "key",
    )
    fields = [f"{key}={config[key]}" for key in ordered_keys if key in config]
    return ", ".join(fields)


def generate_configs(model_spec: ModelSpec, model_class: type, args: argparse.Namespace) -> list[dict[str, Any]]:
    supported = supported_init_params(model_class)

    message_lengths = parse_int_list(args.message_lengths)
    repeats = parse_int_list(args.repeats)
    alphas = parse_float_list(args.alphas)

    local_stats = parse_str_list(args.local_statistics)
    neighborhood_sizes = parse_int_list(args.neighborhood_sizes)
    key_value = parse_key(str(args.key))

    extras: list[dict[str, Any]] = [{}]

    if model_spec.name == "codex" and "local_statistic" in supported:
        extras = [{"local_statistic": stat} for stat in local_stats]

    if model_spec.name == "sonnet" and "neighborhood_size" in supported:
        expanded: list[dict[str, Any]] = []
        for base in extras:
            for size in neighborhood_sizes:
                merged = dict(base)
                merged["neighborhood_size"] = int(size)
                expanded.append(merged)
        extras = expanded

    ecc_options: list[tuple[bool, str | None]] = [(False, None)]
    if "use_error_correction" in supported:
        for mode in model_spec.ecc_modes:
            ecc_options.append((True, mode))

    configs: list[dict[str, Any]] = []
    seen: set[str] = set()

    for msg_len, repeat, alpha, (use_ecc, ecc_mode), extra in itertools.product(
        message_lengths, repeats, alphas, ecc_options, extras
    ):
        config: dict[str, Any] = {
            "message_length": int(msg_len),
            "repeat": int(repeat),
            "alpha": float(alpha),
            "psnr_threshold": float(args.psnr_threshold),
            "max_encode_time": float(args.max_encode_time),
            "max_decode_time": float(args.max_decode_time),
            "key": key_value,
        }

        if "use_error_correction" in supported:
            config["use_error_correction"] = bool(use_ecc)

        if use_ecc and ecc_mode is not None and "ecc_mode" in supported:
            config["ecc_mode"] = str(ecc_mode)

        config.update(extra)
        config = {key: value for key, value in config.items() if key in supported}

        frozen = json.dumps(config, sort_keys=True)
        if frozen in seen:
            continue
        seen.add(frozen)
        configs.append(config)

    if args.max_configs_per_model > 0:
        return configs[: args.max_configs_per_model]
    return configs


def evaluate_config(
    model_name: str,
    model_class: type,
    config: dict[str, Any],
    images: list[tuple[str, np.ndarray]],
    attacks: dict[str, Callable[[np.ndarray, argparse.Namespace], np.ndarray]],
    args: argparse.Namespace,
) -> dict[str, Any]:
    try:
        model = model_class(**config)
    except Exception as exc:
        return {
            "status": "init_error",
            "model": model_name,
            "config": config,
            "error": f"{type(exc).__name__}: {exc}",
        }

    message_length = int(config.get("message_length", getattr(model, "message_length", 32)))

    requested_ecc = bool(config.get("use_error_correction", False))
    ecc_active = requested_ecc
    if requested_ecc and hasattr(model, "_bch"):
        ecc_active = getattr(model, "_bch") is not None
    ecc_mode_requested = str(config.get("ecc_mode", "none")) if requested_ecc else "none"

    encode_times: list[float] = []
    decode_times: list[float] = []
    psnrs: list[float] = []
    family_scores: dict[str, list[float]] = {name: [] for name in ATTACK_NAMES}

    try:
        for image_name, image in images:
            watermark = deterministic_watermark(image_name, message_length, args.seed)

            encode_start = time.perf_counter()
            watermarked = model.encode(image.copy(), watermark.copy())
            encode_times.append(time.perf_counter() - encode_start)

            if not isinstance(watermarked, np.ndarray):
                raise TypeError("encode() must return a numpy.ndarray")

            watermarked = to_uint8_image(ensure_color_image(watermarked))
            if watermarked.shape != image.shape:
                raise ValueError(
                    f"encode() changed shape from {image.shape} to {watermarked.shape}"
                )

            psnrs.append(psnr_db(image, watermarked))

            for family_name in ATTACK_NAMES:
                attacked = attacks[family_name](watermarked, args)
                attacked = ensure_same_layout(attacked, watermarked)

                decode_start = time.perf_counter()
                decoded = model.decode(attacked)
                decode_times.append(time.perf_counter() - decode_start)

                if not isinstance(decoded, np.ndarray):
                    raise TypeError("decode() must return a numpy.ndarray")

                decoded_bits = normalize_bits(decoded, message_length)
                family_scores[family_name].append(symmetric_ber(watermark, decoded_bits))
    except Exception as exc:
        return {
            "status": "runtime_error",
            "model": model_name,
            "config": config,
            "error": f"{type(exc).__name__}: {exc}",
        }

    family_means = {
        f"ber_{family}": float(np.mean(family_scores[family])) if family_scores[family] else float("nan")
        for family in ATTACK_NAMES
    }
    all_scores = [score for family in ATTACK_NAMES for score in family_scores[family]]

    ber_main = float(np.mean(all_scores)) if all_scores else float("nan")
    avg_psnr = float(np.mean(psnrs)) if psnrs else float("nan")

    ber_penalized = 0.5 if avg_psnr < float(args.psnr_threshold) else ber_main

    return {
        "status": "ok",
        "model": model_name,
        "config": config,
        "config_text": compact_config(config),
        "ecc_requested": requested_ecc,
        "ecc_active": bool(ecc_active),
        "ecc_mode_requested": ecc_mode_requested,
        "ecc_profile": ecc_mode_requested,
        "ber_main": ber_main,
        "ber_penalized": ber_penalized,
        "avg_psnr": avg_psnr,
        "encode_time_per_image": float(np.mean(encode_times)) if encode_times else float("nan"),
        "decode_time_per_image": float(np.mean(decode_times)) if decode_times else float("nan"),
        **family_means,
    }


def fmt(value: float, digits: int = 4) -> str:
    if isinstance(value, float):
        if math.isnan(value):
            return "nan"
        if math.isinf(value):
            return "inf"
    return f"{value:.{digits}f}"


def render_table(headers: list[str], rows: list[list[str]]) -> str:
    widths = [len(header) for header in headers]
    for row in rows:
        for index, cell in enumerate(row):
            widths[index] = max(widths[index], len(cell))

    sep = "-+-".join("-" * width for width in widths)
    header_row = " | ".join(header.ljust(widths[i]) for i, header in enumerate(headers))
    data_rows = [" | ".join(cell.ljust(widths[i]) for i, cell in enumerate(row)) for row in rows]
    return "\n".join([header_row, sep, *data_rows])


def sort_key(result: dict[str, Any]) -> tuple[float, float, float, float, float]:
    return (
        float(result["ber_penalized"]),
        float(result["ber_main"]),
        -float(result["avg_psnr"]),
        float(result["encode_time_per_image"]),
        float(result["decode_time_per_image"]),
    )


def save_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return

    field_order = [
        "model",
        "ber_penalized",
        "ber_main",
        "avg_psnr",
        "encode_time_per_image",
        "decode_time_per_image",
        "ecc_requested",
        "ecc_active",
        "ecc_mode_requested",
        "ecc_profile",
        *[f"ber_{name}" for name in ATTACK_NAMES],
        "config_text",
        "config",
        "status",
        "error",
    ]

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=field_order, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            serializable = dict(row)
            config = serializable.get("config")
            if isinstance(config, dict):
                serializable["config"] = json.dumps(config, sort_keys=True)
            writer.writerow(serializable)


def print_summary(valid_results: list[dict[str, Any]], top_k: int) -> None:
    if not valid_results:
        print("No successful configuration to summarize.")
        return

    grouped: dict[str, list[dict[str, Any]]] = {}
    for result in valid_results:
        grouped.setdefault(result["model"], []).append(result)

    best_per_model: list[dict[str, Any]] = []
    for model_name, rows in grouped.items():
        rows.sort(key=sort_key)
        best_per_model.append(rows[0])

    best_per_model.sort(key=sort_key)

    print("\n=== Best Configuration Per Implementation ===")
    headers = [
        "model",
        "BER penalized",
        "BER main",
        "PSNR",
        "enc s/img",
        "dec s/img",
        "ECC active",
        "config",
    ]
    rows = [
        [
            result["model"],
            fmt(result["ber_penalized"]),
            fmt(result["ber_main"]),
            fmt(result["avg_psnr"], 2),
            fmt(result["encode_time_per_image"], 4),
            fmt(result["decode_time_per_image"], 4),
            str(result["ecc_active"]),
            result["config_text"],
        ]
        for result in best_per_model
    ]
    print(render_table(headers, rows))

    print("\n=== BER By Attack Family (Best Config Per Model) ===")
    attack_headers = ["model", *ATTACK_NAMES]
    attack_rows = []
    for result in best_per_model:
        attack_rows.append([result["model"], *[fmt(result[f"ber_{name}"]) for name in ATTACK_NAMES]])
    print(render_table(attack_headers, attack_rows))

    print("\n=== Best Configuration By ECC Profile ===")
    ecc_headers = [
        "model",
        "ECC profile",
        "BER penalized",
        "PSNR",
        "enc s/img",
        "dec s/img",
        "ECC active",
        "config",
    ]
    ecc_rows: list[list[str]] = []
    for model_name in sorted(grouped):
        rows = grouped[model_name]
        by_profile: dict[str, list[dict[str, Any]]] = {}
        for row in rows:
            by_profile.setdefault(str(row.get("ecc_profile", "none")), []).append(row)
        for profile_name in sorted(by_profile):
            best = sorted(by_profile[profile_name], key=sort_key)[0]
            ecc_rows.append(
                [
                    model_name,
                    profile_name,
                    fmt(best["ber_penalized"]),
                    fmt(best["avg_psnr"], 2),
                    fmt(best["encode_time_per_image"], 4),
                    fmt(best["decode_time_per_image"], 4),
                    str(best["ecc_active"]),
                    best["config_text"],
                ]
            )
    print(render_table(ecc_headers, ecc_rows))

    for model_name in sorted(grouped):
        rows = sorted(grouped[model_name], key=sort_key)[:top_k]
        print(f"\n=== Top {top_k} Configs: {model_name} ===")
        top_headers = ["rank", "BER pen.", "PSNR", "enc s/img", "dec s/img", "config"]
        top_rows = []
        for rank, result in enumerate(rows, start=1):
            top_rows.append(
                [
                    str(rank),
                    fmt(result["ber_penalized"]),
                    fmt(result["avg_psnr"], 2),
                    fmt(result["encode_time_per_image"], 4),
                    fmt(result["decode_time_per_image"], 4),
                    result["config_text"],
                ]
            )
        print(render_table(top_headers, top_rows))


def main() -> int:
    args = parse_args()

    selected_model_names = parse_str_list(args.models)
    selected_specs = resolve_model_specs(selected_model_names)

    images_dir = Path(args.images_dir).resolve()
    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = (TESTS_DIR / out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    images = load_images(images_dir)
    attacks = build_attacks(args)

    print(f"Loaded {len(images)} image(s) from {images_dir}")
    print(f"Selected implementation(s): {', '.join(spec.name for spec in selected_specs)}")

    all_results: list[dict[str, Any]] = []

    for model_spec in selected_specs:
        model_class = load_watermark_class(model_spec)
        configs = generate_configs(model_spec, model_class, args)

        print(f"\n[{model_spec.name}] evaluating {len(configs)} configuration(s)...")

        for index, config in enumerate(configs, start=1):
            if not args.quiet and (index == 1 or index % 10 == 0 or index == len(configs)):
                print(f"  -> {model_spec.name}: {index}/{len(configs)}")

            result = evaluate_config(model_spec.name, model_class, config, images, attacks, args)
            all_results.append(result)

    valid_results = [result for result in all_results if result["status"] == "ok"]
    failed_results = [result for result in all_results if result["status"] != "ok"]

    valid_results.sort(key=sort_key)

    print("\n=== Benchmark Summary ===")
    print(f"Successful configs: {len(valid_results)}")
    print(f"Failed configs: {len(failed_results)}")

    if failed_results:
        print("\nExamples of failed configs:")
        for failed in failed_results[:5]:
            print(f"- {failed['model']}: {failed.get('error', 'unknown error')} | {compact_config(failed['config'])}")

    print_summary(valid_results, top_k=max(1, int(args.top_k)))

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    all_csv = out_dir / f"benchmark_all_{timestamp}.csv"
    valid_csv = out_dir / f"benchmark_valid_{timestamp}.csv"
    json_report = out_dir / f"benchmark_report_{timestamp}.json"

    save_csv(all_csv, all_results)
    save_csv(valid_csv, valid_results)

    with json_report.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "generated_at": timestamp,
                "settings": vars(args),
                "valid_count": len(valid_results),
                "failed_count": len(failed_results),
                "best_global": valid_results[0] if valid_results else None,
                "results": all_results,
            },
            handle,
            indent=2,
        )

    print("\nSaved reports:")
    print(f"- {all_csv}")
    print(f"- {valid_csv}")
    print(f"- {json_report}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
