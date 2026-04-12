"""
Benchmark parameter sweeps for the single WatermarkModel implementation.

This script benchmarks LSB/model.py only.

Key capabilities
----------------
1. Parameter sweeps for core watermark settings:
   message_length, alpha, use_crc, msg_repeat, crc_repeat,
   neighborhood_size, max_flip_bits, key.
2. Parameter sweeps for the NVF method (when supported by the model):
   use_nvf, nvf_window_size, nvf_D, nvf_alpha_low.
3. Built-in default efficiency check:
   compares default parameters with use_nvf=False vs use_nvf=True.
4. Attack-aware scoring and reporting:
   BER, worst-attack BER, PSNR, encode/decode runtime.

Images are loaded recursively from LSB/img with paths containing "_old"
explicitly excluded.

Example
-------
uv run --directory LSB python tests/compare_watermark_models.py --top-k 12
"""

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
from pathlib import Path
from typing import Any, Callable, Optional

import cv2
import numpy as np


ROOT_DIR = Path(__file__).resolve().parents[1]
TESTS_DIR = Path(__file__).resolve().parent
MODEL_PATH = ROOT_DIR / "model.py"
CRC_BITS = 32


ATTACK_NAMES = (
    "none",
    "jpeg",
    "blur",
    "gaussian_noise",
    "rotation",
    "brightness",
    "contrast",
    "resize",
)


def parse_int_list(value: str) -> list[int]:
    values = [int(tok.strip()) for tok in value.split(",") if tok.strip()]
    if not values:
        raise ValueError("Expected at least one integer value.")
    return values


def parse_float_list(value: str) -> list[float]:
    values = [float(tok.strip()) for tok in value.split(",") if tok.strip()]
    if not values:
        raise ValueError("Expected at least one float value.")
    return values


def parse_str_list(value: str) -> list[str]:
    values = [tok.strip() for tok in value.split(",") if tok.strip()]
    if not values:
        raise ValueError("Expected at least one string value.")
    return values


def parse_bool(token: str) -> bool:
    lowered = token.strip().lower()
    if lowered in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if lowered in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise ValueError(f"Invalid boolean token: {token!r}")


def parse_bool_list(value: str) -> list[bool]:
    values = [parse_bool(tok) for tok in value.split(",") if tok.strip()]
    if not values:
        raise ValueError("Expected at least one boolean value.")
    return values


def parse_optional_float_token(token: str) -> Optional[float]:
    lowered = token.strip().lower()
    if lowered in {"auto", "none", "null", "default"}:
        return None
    return float(token)


def parse_optional_float_list(value: str) -> list[Optional[float]]:
    values = [parse_optional_float_token(tok) for tok in value.split(",") if tok.strip()]
    if not values:
        raise ValueError("Expected at least one optional-float token.")
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark parameter combinations for LSB/model.py WatermarkModel "
            "including optional NVF settings, with BER/PSNR/runtime metrics."
        )
    )

    parser.add_argument(
        "--images-dir",
        default=str(ROOT_DIR / "img"),
        help="Image directory. Loaded recursively; folders named '_old' are skipped.",
    )
    parser.add_argument(
        "--out-dir",
        default="results",
        help="Output directory for benchmark CSV/JSON reports.",
    )

    parser.add_argument(
        "--message-lengths",
        default="32,64",
        help="Comma-separated message_length values.",
    )
    parser.add_argument(
        "--alphas",
        default="75,85,95",
        help="Comma-separated alpha values.",
    )
    parser.add_argument(
        "--use-crc",
        default="false,true",
        help="Comma-separated booleans for use_crc.",
    )
    parser.add_argument(
        "--msg-repeats",
        default="20,35,50",
        help="Comma-separated msg_repeat values.",
    )
    parser.add_argument(
        "--crc-repeats",
        default="25,40,55",
        help="Comma-separated crc_repeat values.",
    )
    parser.add_argument(
        "--neighborhood-sizes",
        default="3,5",
        help="Comma-separated odd neighborhood_size values >= 3.",
    )
    parser.add_argument(
        "--max-flip-bits",
        default="2",
        help="Comma-separated max_flip_bits values.",
    )
    parser.add_argument("--key", type=int, default=42, help="Model key seed.")

    parser.add_argument(
        "--use-nvf",
        default="false,true",
        help="Comma-separated booleans for use_nvf.",
    )
    parser.add_argument(
        "--nvf-window-sizes",
        default="3",
        help="Comma-separated odd nvf_window_size values.",
    )
    parser.add_argument(
        "--nvf-d-values",
        default="50",
        help="Comma-separated nvf_D values.",
    )
    parser.add_argument(
        "--nvf-alpha-lows",
        default="auto",
        help="Comma-separated nvf_alpha_low values or auto.",
    )

    parser.add_argument(
        "--attacks",
        default="none,jpeg,blur,gaussian_noise,rotation",
        help="Comma-separated attack names. Available: " + ", ".join(ATTACK_NAMES),
    )
    parser.add_argument("--jpeg-quality", type=int, default=55, help="JPEG quality in [1, 100].")
    parser.add_argument("--blur-kernel", type=int, default=3, help="Odd Gaussian blur kernel size.")
    parser.add_argument("--noise-sigma", type=float, default=4.0, help="Gaussian noise sigma.")
    parser.add_argument("--rotation-degrees", type=float, default=8.0, help="Rotation attack angle in degrees.")
    parser.add_argument("--brightness-shift", type=float, default=18.0, help="Brightness beta shift.")
    parser.add_argument("--contrast-factor", type=float, default=1.2, help="Contrast alpha factor.")
    parser.add_argument(
        "--resize-scale",
        type=float,
        default=0.5,
        help="Downscale factor used by resize attack before upscaling.",
    )

    parser.add_argument("--seed", type=int, default=12345, help="Global deterministic seed.")
    parser.add_argument("--psnr-target", type=float, default=30.0, help="Target PSNR for quality penalty.")
    parser.add_argument(
        "--psnr-penalty-weight",
        type=float,
        default=0.25,
        help="Penalty multiplier applied when avg PSNR falls below target.",
    )
    parser.add_argument("--top-k", type=int, default=12, help="How many top configs to print.")
    parser.add_argument(
        "--max-configs",
        type=int,
        default=0,
        help="If > 0, evaluate only the first N generated configurations.",
    )
    parser.add_argument(
        "--skip-default-efficiency-check",
        action="store_true",
        help="Disable default use_nvf=False vs use_nvf=True comparison.",
    )
    parser.add_argument("--quiet", action="store_true", help="Reduce progress logging.")

    return parser.parse_args()


def stable_seed(*tokens: Any) -> int:
    joined = "|".join(str(tok) for tok in tokens)
    digest = hashlib.blake2b(joined.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "little", signed=False)


def load_watermark_class(model_path: Path) -> type:
    if not model_path.exists():
        raise FileNotFoundError(f"Missing model file: {model_path}")

    module_name = f"benchmark_model_{stable_seed(model_path)}"
    module_spec = importlib.util.spec_from_file_location(module_name, model_path)
    if module_spec is None or module_spec.loader is None:
        raise ImportError(f"Unable to load module from {model_path}")

    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)

    if not hasattr(module, "WatermarkModel"):
        raise AttributeError(f"{model_path} does not define WatermarkModel")

    watermark_class = getattr(module, "WatermarkModel")
    if not inspect.isclass(watermark_class):
        raise TypeError("WatermarkModel is not a class")

    for required in ("encode", "decode"):
        method = getattr(watermark_class, required, None)
        if method is None or not callable(method):
            raise TypeError(f"WatermarkModel missing callable {required}()")

    return watermark_class


def supported_init_params(model_class: type) -> set[str]:
    signature = inspect.signature(model_class.__init__)
    params: set[str] = set()
    for name, param in signature.parameters.items():
        if name == "self":
            continue
        if param.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY):
            params.add(name)
    return params


def default_constructor_config(model_class: type) -> dict[str, Any]:
    signature = inspect.signature(model_class.__init__)
    defaults: dict[str, Any] = {}
    for name, param in signature.parameters.items():
        if name == "self":
            continue
        if param.kind not in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY):
            continue
        if param.default is inspect.Parameter.empty:
            continue
        defaults[name] = param.default
    return defaults


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

    suffixes = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}
    image_paths: list[Path] = []

    for path in images_dir.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix.lower() not in suffixes:
            continue
        if any(part.lower() == "_old" for part in path.parts):
            continue
        image_paths.append(path)

    image_paths = sorted(set(image_paths))
    if not image_paths:
        raise RuntimeError(f"No supported images found in {images_dir}")

    loaded: list[tuple[str, np.ndarray]] = []
    for image_path in image_paths:
        img = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if img is None:
            continue
        relative_name = str(image_path.relative_to(images_dir)).replace("\\", "/")
        loaded.append((relative_name, to_uint8_image(ensure_color_image(img))))

    if not loaded:
        raise RuntimeError(f"Could not read any image from {images_dir}")
    return loaded


def psnr_db(reference: np.ndarray, compared: np.ndarray) -> float:
    ref = reference.astype(np.float32, copy=False)
    cmp = compared.astype(np.float32, copy=False)
    mse = float(np.mean((ref - cmp) ** 2, dtype=np.float64))
    if mse <= 0.0:
        return float("inf")
    return float(10.0 * np.log10((255.0 * 255.0) / mse))


def deterministic_watermark(image_name: str, message_length: int, global_seed: int) -> np.ndarray:
    rng = np.random.default_rng(stable_seed(global_seed, image_name, message_length))
    return rng.integers(0, 2, size=message_length, dtype=np.uint8)


def build_watermark_map(
    images: list[tuple[str, np.ndarray]],
    message_length: int,
    seed: int,
) -> dict[str, np.ndarray]:
    return {
        image_name: deterministic_watermark(image_name, message_length, seed)
        for image_name, _ in images
    }


def normalize_bits(bits: np.ndarray, length: int) -> np.ndarray:
    arr = np.asarray(bits).reshape(-1)
    out = np.zeros(length, dtype=np.uint8)
    if arr.size == 0:
        return out
    clipped = arr[:length]
    out[: clipped.size] = (clipped > 0).astype(np.uint8, copy=False)
    return out


def bit_error_rate(reference_bits: np.ndarray, decoded_bits: np.ndarray) -> float:
    if reference_bits.size == 0:
        return 0.0
    return float(np.mean(reference_bits != decoded_bits))


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


AttackFn = Callable[[np.ndarray, argparse.Namespace, np.random.Generator], np.ndarray]


def attack_none(image: np.ndarray, _: argparse.Namespace, __: np.random.Generator) -> np.ndarray:
    return image.copy()


def attack_jpeg(image: np.ndarray, args: argparse.Namespace, __: np.random.Generator) -> np.ndarray:
    quality = int(np.clip(args.jpeg_quality, 1, 100))
    ok, encoded = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        return image.copy()
    decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    return image.copy() if decoded is None else decoded


def attack_blur(image: np.ndarray, args: argparse.Namespace, __: np.random.Generator) -> np.ndarray:
    kernel = max(1, int(args.blur_kernel))
    if kernel % 2 == 0:
        kernel += 1
    return cv2.GaussianBlur(image, (kernel, kernel), sigmaX=0.0)


def attack_gaussian_noise(image: np.ndarray, args: argparse.Namespace, rng: np.random.Generator) -> np.ndarray:
    sigma = max(0.0, float(args.noise_sigma))
    if sigma == 0.0:
        return image.copy()
    noise = rng.normal(loc=0.0, scale=sigma, size=image.shape).astype(np.float32)
    noisy = image.astype(np.float32, copy=False) + noise
    return np.clip(noisy, 0.0, 255.0).astype(np.uint8)


def attack_rotation(image: np.ndarray, args: argparse.Namespace, __: np.random.Generator) -> np.ndarray:
    h, w = image.shape[:2]
    matrix = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), float(args.rotation_degrees), 1.0)
    return cv2.warpAffine(
        image,
        matrix,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101,
    )


def attack_brightness(image: np.ndarray, args: argparse.Namespace, __: np.random.Generator) -> np.ndarray:
    return cv2.convertScaleAbs(image, alpha=1.0, beta=float(args.brightness_shift))


def attack_contrast(image: np.ndarray, args: argparse.Namespace, __: np.random.Generator) -> np.ndarray:
    return cv2.convertScaleAbs(image, alpha=float(args.contrast_factor), beta=0.0)


def attack_resize(image: np.ndarray, args: argparse.Namespace, __: np.random.Generator) -> np.ndarray:
    scale = float(args.resize_scale)
    scale = min(max(scale, 0.1), 1.0)
    if math.isclose(scale, 1.0, rel_tol=0.0, abs_tol=1e-9):
        return image.copy()

    h, w = image.shape[:2]
    tw = max(1, int(round(w * scale)))
    th = max(1, int(round(h * scale)))
    down = cv2.resize(image, (tw, th), interpolation=cv2.INTER_AREA)
    return cv2.resize(down, (w, h), interpolation=cv2.INTER_LINEAR)


def build_attacks(selected_names: list[str]) -> dict[str, AttackFn]:
    registry: dict[str, AttackFn] = {
        "none": attack_none,
        "jpeg": attack_jpeg,
        "blur": attack_blur,
        "gaussian_noise": attack_gaussian_noise,
        "rotation": attack_rotation,
        "brightness": attack_brightness,
        "contrast": attack_contrast,
        "resize": attack_resize,
    }

    unknown = [name for name in selected_names if name not in registry]
    if unknown:
        raise ValueError(
            "Unknown attack(s): "
            + ", ".join(unknown)
            + ". Valid values: "
            + ", ".join(sorted(registry))
        )

    deduped = list(dict.fromkeys(selected_names))
    return {name: registry[name] for name in deduped}


def config_text(config: dict[str, Any]) -> str:
    ordered = (
        "message_length",
        "alpha",
        "use_crc",
        "msg_repeat",
        "crc_repeat",
        "neighborhood_size",
        "max_flip_bits",
        "use_nvf",
        "nvf_window_size",
        "nvf_D",
        "nvf_alpha_low",
        "key",
    )
    return ", ".join(f"{key}={config[key]}" for key in ordered if key in config)


def payload_bits(message_length: int, msg_repeat: int, use_crc: bool, crc_repeat: int) -> int:
    if use_crc:
        return message_length * msg_repeat + CRC_BITS * crc_repeat
    return message_length * msg_repeat


def config_payload_bits(config: dict[str, Any]) -> int:
    message_length = int(config.get("message_length", 32))
    msg_repeat = int(config.get("msg_repeat", 1))
    use_crc = bool(config.get("use_crc", False))
    crc_repeat = int(config.get("crc_repeat", 1))
    return payload_bits(message_length, msg_repeat, use_crc, crc_repeat)


def config_fits_capacity(config: dict[str, Any], min_capacity: int) -> bool:
    return config_payload_bits(config) <= min_capacity


def generate_configs(
    args: argparse.Namespace,
    supported: set[str],
    min_capacity: int,
) -> list[dict[str, Any]]:
    has = lambda name: name in supported

    message_lengths = parse_int_list(args.message_lengths) if has("message_length") else [32]
    alphas = parse_float_list(args.alphas) if has("alpha") else [10.0]
    use_crc_values = parse_bool_list(args.use_crc) if has("use_crc") else [False]
    msg_repeats = parse_int_list(args.msg_repeats) if has("msg_repeat") else [1]
    crc_repeats = parse_int_list(args.crc_repeats) if has("crc_repeat") else [1]
    neighborhood_sizes = (
        parse_int_list(args.neighborhood_sizes) if has("neighborhood_size") else [3]
    )
    max_flip_values = parse_int_list(args.max_flip_bits) if has("max_flip_bits") else [0]
    use_nvf_values = parse_bool_list(args.use_nvf) if has("use_nvf") else [False]
    nvf_window_values = parse_int_list(args.nvf_window_sizes) if has("nvf_window_size") else [3]
    nvf_d_values = parse_float_list(args.nvf_d_values) if has("nvf_D") else [50.0]
    nvf_alpha_low_values = (
        parse_optional_float_list(args.nvf_alpha_lows) if has("nvf_alpha_low") else [None]
    )

    configs: list[dict[str, Any]] = []
    seen: set[str] = set()

    for (
        msg_len,
        alpha,
        use_crc,
        msg_repeat,
        neighborhood_size,
        max_flip_bits,
        use_nvf,
    ) in itertools.product(
        message_lengths,
        alphas,
        use_crc_values,
        msg_repeats,
        neighborhood_sizes,
        max_flip_values,
        use_nvf_values,
    ):
        if msg_len <= 0 or msg_repeat <= 0:
            continue
        if neighborhood_size < 3 or neighborhood_size % 2 == 0:
            continue
        if max_flip_bits < 0:
            continue

        crc_candidates = crc_repeats if use_crc else [crc_repeats[0]]
        nvf_iter = (
            itertools.product(nvf_window_values, nvf_d_values, nvf_alpha_low_values)
            if use_nvf
            else [(nvf_window_values[0], nvf_d_values[0], nvf_alpha_low_values[0])]
        )

        for crc_repeat in crc_candidates:
            if crc_repeat <= 0:
                continue
            if use_crc and crc_repeat <= msg_repeat:
                continue

            for nvf_window_size, nvf_d, nvf_alpha_low in nvf_iter:
                if nvf_window_size < 1 or nvf_window_size % 2 == 0:
                    continue
                if nvf_d <= 0:
                    continue
                if use_nvf and nvf_alpha_low is not None and float(nvf_alpha_low) >= float(alpha):
                    continue

                candidate: dict[str, Any] = {
                    "message_length": int(msg_len),
                    "alpha": float(alpha),
                    "use_crc": bool(use_crc),
                    "msg_repeat": int(msg_repeat),
                    "crc_repeat": int(crc_repeat),
                    "neighborhood_size": int(neighborhood_size),
                    "max_flip_bits": int(max_flip_bits),
                    "use_nvf": bool(use_nvf),
                    "nvf_window_size": int(nvf_window_size),
                    "nvf_D": float(nvf_d),
                    "nvf_alpha_low": None if nvf_alpha_low is None else float(nvf_alpha_low),
                    "key": int(args.key),
                }

                candidate = {key: value for key, value in candidate.items() if key in supported}
                if not config_fits_capacity(candidate, min_capacity):
                    continue

                frozen = json.dumps(candidate, sort_keys=True)
                if frozen in seen:
                    continue

                seen.add(frozen)
                configs.append(candidate)

    if args.max_configs > 0:
        return configs[: args.max_configs]
    return configs


def evaluate_config(
    model_class: type,
    config: dict[str, Any],
    images: list[tuple[str, np.ndarray]],
    watermarks_by_image: dict[str, np.ndarray],
    attacks: dict[str, AttackFn],
    args: argparse.Namespace,
) -> dict[str, Any]:
    try:
        model = model_class(**config)
    except Exception as exc:
        return {
            "status": "init_error",
            "config": config,
            "error": f"{type(exc).__name__}: {exc}",
        }

    message_length = int(config.get("message_length", 32))
    config_token = json.dumps(config, sort_keys=True)

    encode_times: list[float] = []
    decode_times: list[float] = []
    psnr_values: list[float] = []
    attack_bers: dict[str, list[float]] = {name: [] for name in attacks}

    try:
        for image_name, image in images:
            watermark = watermarks_by_image[image_name]

            encode_start = time.perf_counter()
            watermarked = model.encode(image.copy(), watermark.copy())
            encode_times.append(time.perf_counter() - encode_start)

            if not isinstance(watermarked, np.ndarray):
                raise TypeError("encode() must return a numpy.ndarray")

            watermarked = to_uint8_image(ensure_color_image(watermarked))
            if watermarked.shape != image.shape:
                raise ValueError(f"encode() changed image shape {image.shape} -> {watermarked.shape}")

            psnr_values.append(psnr_db(image, watermarked))

            for attack_name, attack_fn in attacks.items():
                attack_rng = np.random.default_rng(
                    stable_seed(args.seed, config_token, image_name, attack_name)
                )
                attacked = attack_fn(watermarked, args, attack_rng)
                attacked = ensure_same_layout(attacked, watermarked)

                decode_start = time.perf_counter()
                decoded = model.decode(attacked)
                decode_times.append(time.perf_counter() - decode_start)

                if not isinstance(decoded, np.ndarray):
                    raise TypeError("decode() must return a numpy.ndarray")

                decoded_bits = normalize_bits(decoded, message_length)
                attack_bers[attack_name].append(bit_error_rate(watermark, decoded_bits))
    except Exception as exc:
        return {
            "status": "runtime_error",
            "config": config,
            "error": f"{type(exc).__name__}: {exc}",
        }

    attack_means = {
        f"ber_{name}": float(np.mean(values)) if values else float("nan")
        for name, values in attack_bers.items()
    }
    all_bers = [ber for values in attack_bers.values() for ber in values]

    ber_mean = float(np.mean(all_bers)) if all_bers else float("nan")
    avg_psnr = float(np.mean(psnr_values)) if psnr_values else float("nan")
    ber_worst_attack = float(np.nanmax(list(attack_means.values()))) if attack_means else float("nan")

    if math.isfinite(avg_psnr) and args.psnr_target > 0:
        psnr_penalty = max(0.0, (args.psnr_target - avg_psnr) / args.psnr_target)
    else:
        psnr_penalty = 0.0

    score = ber_mean + float(args.psnr_penalty_weight) * psnr_penalty

    return {
        "status": "ok",
        "config": config,
        "config_text": config_text(config),
        "score": score,
        "ber_mean": ber_mean,
        "ber_worst_attack": ber_worst_attack,
        "avg_psnr": avg_psnr,
        "encode_time_per_image": float(np.mean(encode_times)) if encode_times else float("nan"),
        "decode_time_per_call": float(np.mean(decode_times)) if decode_times else float("nan"),
        **attack_means,
    }


def sort_key(result: dict[str, Any]) -> tuple[float, float, float, float, float, float]:
    return (
        float(result["score"]),
        float(result["ber_mean"]),
        float(result["ber_worst_attack"]),
        -float(result["avg_psnr"]),
        float(result["encode_time_per_image"]),
        float(result["decode_time_per_call"]),
    )


def fmt(value: float, digits: int = 4) -> str:
    if math.isnan(value):
        return "nan"
    if math.isinf(value):
        return "inf"
    return f"{value:.{digits}f}"


def render_table(headers: list[str], rows: list[list[str]]) -> str:
    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))

    sep = "-+-".join("-" * width for width in widths)
    header_row = " | ".join(headers[i].ljust(widths[i]) for i in range(len(headers)))
    body_rows = [" | ".join(row[i].ljust(widths[i]) for i in range(len(headers))) for row in rows]
    return "\n".join([header_row, sep, *body_rows])


def save_csv(path: Path, rows: list[dict[str, Any]], attack_names: list[str]) -> None:
    if not rows:
        return

    columns = [
        "status",
        "score",
        "ber_mean",
        "ber_worst_attack",
        "avg_psnr",
        "encode_time_per_image",
        "decode_time_per_call",
        *[f"ber_{name}" for name in attack_names],
        "config_text",
        "config",
        "error",
    ]

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            item = dict(row)
            config = item.get("config")
            if isinstance(config, dict):
                item["config"] = json.dumps(config, sort_keys=True)
            writer.writerow(item)


def print_summary(valid_results: list[dict[str, Any]], attack_names: list[str], top_k: int) -> None:
    if not valid_results:
        print("No successful configuration to summarize.")
        return

    best = valid_results[0]
    print("\n=== Best Configuration ===")
    print(best["config_text"])
    print(
        "score={score}, ber_mean={ber}, ber_worst_attack={worst}, psnr={psnr}, enc_s/img={enc}, dec_s/call={dec}".format(
            score=fmt(best["score"]),
            ber=fmt(best["ber_mean"]),
            worst=fmt(best["ber_worst_attack"]),
            psnr=fmt(best["avg_psnr"], 2),
            enc=fmt(best["encode_time_per_image"], 5),
            dec=fmt(best["decode_time_per_call"], 5),
        )
    )

    print("\n=== BER By Attack (Best Config) ===")
    ber_headers = ["attack", "ber"]
    ber_rows = [[attack, fmt(best.get(f"ber_{attack}", float("nan")))] for attack in attack_names]
    print(render_table(ber_headers, ber_rows))

    top_n = max(1, int(top_k))
    print(f"\n=== Top {top_n} Configurations ===")
    headers = ["rank", "score", "ber", "worst", "psnr", "enc s/img", "dec s/call", "config"]
    rows: list[list[str]] = []

    for rank, result in enumerate(valid_results[:top_n], start=1):
        rows.append(
            [
                str(rank),
                fmt(result["score"]),
                fmt(result["ber_mean"]),
                fmt(result["ber_worst_attack"]),
                fmt(result["avg_psnr"], 2),
                fmt(result["encode_time_per_image"], 5),
                fmt(result["decode_time_per_call"], 5),
                result["config_text"],
            ]
        )
    print(render_table(headers, rows))


def evaluate_default_nvf_efficiency(
    model_class: type,
    supported: set[str],
    images: list[tuple[str, np.ndarray]],
    attacks: dict[str, AttackFn],
    args: argparse.Namespace,
    min_capacity: int,
    watermark_cache: dict[int, dict[str, np.ndarray]],
) -> dict[str, Any]:
    if "use_nvf" not in supported:
        return {
            "status": "skipped",
            "reason": "use_nvf is not supported by WatermarkModel.",
        }

    defaults = default_constructor_config(model_class)
    if not defaults:
        return {
            "status": "skipped",
            "reason": "Could not extract default constructor parameters.",
        }

    base = {key: value for key, value in defaults.items() if key in supported}
    config_no_nvf = dict(base)
    config_no_nvf["use_nvf"] = False
    config_with_nvf = dict(base)
    config_with_nvf["use_nvf"] = True

    if not config_fits_capacity(config_no_nvf, min_capacity) or not config_fits_capacity(config_with_nvf, min_capacity):
        return {
            "status": "skipped",
            "reason": "Default NVF comparison does not fit image payload capacity.",
            "default_no_nvf": config_no_nvf,
            "default_with_nvf": config_with_nvf,
        }

    msg_len = int(config_no_nvf.get("message_length", 32))
    if msg_len not in watermark_cache:
        watermark_cache[msg_len] = build_watermark_map(images, msg_len, args.seed)
    watermarks = watermark_cache[msg_len]

    result_no_nvf = evaluate_config(model_class, config_no_nvf, images, watermarks, attacks, args)
    result_with_nvf = evaluate_config(model_class, config_with_nvf, images, watermarks, attacks, args)

    out: dict[str, Any] = {
        "status": "ok",
        "default_no_nvf": result_no_nvf,
        "default_with_nvf": result_with_nvf,
    }

    if result_no_nvf.get("status") == "ok" and result_with_nvf.get("status") == "ok":
        total_runtime_no_nvf = float(result_no_nvf["encode_time_per_image"]) + float(result_no_nvf["decode_time_per_call"])
        total_runtime_with_nvf = float(result_with_nvf["encode_time_per_image"]) + float(result_with_nvf["decode_time_per_call"])

        out["delta_with_minus_without"] = {
            "score": float(result_with_nvf["score"]) - float(result_no_nvf["score"]),
            "ber_mean": float(result_with_nvf["ber_mean"]) - float(result_no_nvf["ber_mean"]),
            "ber_worst_attack": float(result_with_nvf["ber_worst_attack"]) - float(result_no_nvf["ber_worst_attack"]),
            "avg_psnr": float(result_with_nvf["avg_psnr"]) - float(result_no_nvf["avg_psnr"]),
            "encode_time_per_image": float(result_with_nvf["encode_time_per_image"]) - float(result_no_nvf["encode_time_per_image"]),
            "decode_time_per_call": float(result_with_nvf["decode_time_per_call"]) - float(result_no_nvf["decode_time_per_call"]),
            "total_runtime": total_runtime_with_nvf - total_runtime_no_nvf,
        }

    return out


def print_default_nvf_efficiency_report(report: dict[str, Any]) -> None:
    print("\n=== Default NVF Efficiency Check ===")

    if report.get("status") != "ok":
        print(f"Skipped: {report.get('reason', 'unknown reason')}")
        return

    no_nvf = report.get("default_no_nvf", {})
    with_nvf = report.get("default_with_nvf", {})

    if no_nvf.get("status") != "ok" or with_nvf.get("status") != "ok":
        print("Could not complete default comparison due to runtime/init errors.")
        print(f"- no_nvf status: {no_nvf.get('status')} | error: {no_nvf.get('error', 'none')}")
        print(f"- with_nvf status: {with_nvf.get('status')} | error: {with_nvf.get('error', 'none')}")
        return

    headers = ["profile", "score", "ber", "worst", "psnr", "enc s/img", "dec s/call"]
    rows = [
        [
            "default use_nvf=False",
            fmt(float(no_nvf["score"])),
            fmt(float(no_nvf["ber_mean"])),
            fmt(float(no_nvf["ber_worst_attack"])),
            fmt(float(no_nvf["avg_psnr"]), 2),
            fmt(float(no_nvf["encode_time_per_image"]), 5),
            fmt(float(no_nvf["decode_time_per_call"]), 5),
        ],
        [
            "default use_nvf=True",
            fmt(float(with_nvf["score"])),
            fmt(float(with_nvf["ber_mean"])),
            fmt(float(with_nvf["ber_worst_attack"])),
            fmt(float(with_nvf["avg_psnr"]), 2),
            fmt(float(with_nvf["encode_time_per_image"]), 5),
            fmt(float(with_nvf["decode_time_per_call"]), 5),
        ],
    ]
    print(render_table(headers, rows))

    delta = report.get("delta_with_minus_without")
    if not isinstance(delta, dict):
        return

    print("\nDelta (use_nvf=True minus use_nvf=False):")
    print(
        "score={score}, ber={ber}, worst={worst}, psnr={psnr}, total_runtime={runtime}".format(
            score=fmt(float(delta["score"])),
            ber=fmt(float(delta["ber_mean"])),
            worst=fmt(float(delta["ber_worst_attack"])),
            psnr=fmt(float(delta["avg_psnr"]), 2),
            runtime=fmt(float(delta["total_runtime"]), 5),
        )
    )

    improves_quality = float(delta["score"]) < 0.0
    faster_runtime = float(delta["total_runtime"]) < 0.0

    if improves_quality and faster_runtime:
        verdict = "NVF is more efficient overall for default settings (better score and faster runtime)."
    elif improves_quality and not faster_runtime:
        verdict = "NVF improves robustness-quality score at default settings, but increases runtime."
    elif (not improves_quality) and faster_runtime:
        verdict = "NVF is faster at default settings, but worsens robustness-quality score."
    else:
        verdict = "NVF is less efficient at default settings for both score and runtime."
    print(f"Verdict: {verdict}")


def main() -> int:
    args = parse_args()

    model_class = load_watermark_class(MODEL_PATH)
    supported = supported_init_params(model_class)

    images_dir = Path(args.images_dir).resolve()
    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = (TESTS_DIR / out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    images = load_images(images_dir)
    attack_names = parse_str_list(args.attacks)
    attacks = build_attacks(attack_names)

    min_capacity = min(img.shape[0] * img.shape[1] * img.shape[2] for _, img in images)
    configs = generate_configs(args, supported, int(min_capacity))
    if not configs:
        raise RuntimeError("No valid configurations generated. Check parameter ranges.")

    unique_lengths = sorted({int(cfg.get("message_length", 32)) for cfg in configs})
    watermark_cache: dict[int, dict[str, np.ndarray]] = {
        message_length: build_watermark_map(images, message_length, args.seed)
        for message_length in unique_lengths
    }

    print(f"Model file: {MODEL_PATH}")
    print(f"Supported init params: {', '.join(sorted(supported))}")
    print(f"Loaded {len(images)} image(s) from {images_dir} (excluding _old)")
    print(f"Attacks: {', '.join(attacks.keys())}")
    print(f"Generated {len(configs)} configuration(s)")

    all_results: list[dict[str, Any]] = []

    for idx, config in enumerate(configs, start=1):
        if not args.quiet and (idx == 1 or idx % 10 == 0 or idx == len(configs)):
            print(f"Evaluating config {idx}/{len(configs)}")

        message_length = int(config.get("message_length", 32))
        result = evaluate_config(
            model_class,
            config,
            images,
            watermark_cache[message_length],
            attacks,
            args,
        )
        all_results.append(result)

    valid_results = [row for row in all_results if row.get("status") == "ok"]
    failed_results = [row for row in all_results if row.get("status") != "ok"]
    valid_results.sort(key=sort_key)

    print("\n=== Benchmark Summary ===")
    print(f"Successful configs: {len(valid_results)}")
    print(f"Failed configs: {len(failed_results)}")

    if failed_results:
        print("\nExample failures:")
        for failed in failed_results[:5]:
            text = config_text(failed.get("config", {}))
            print(f"- {failed.get('status', 'error')}: {failed.get('error', 'unknown error')} | {text}")

    print_summary(valid_results, attack_names, top_k=max(1, int(args.top_k)))

    default_nvf_report: dict[str, Any] | None = None
    if args.skip_default_efficiency_check:
        print("\n=== Default NVF Efficiency Check ===")
        print("Skipped by --skip-default-efficiency-check")
    else:
        default_nvf_report = evaluate_default_nvf_efficiency(
            model_class,
            supported,
            images,
            attacks,
            args,
            int(min_capacity),
            watermark_cache,
        )
        print_default_nvf_efficiency_report(default_nvf_report)

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    all_csv = out_dir / f"benchmark_all_{timestamp}.csv"
    valid_csv = out_dir / f"benchmark_valid_{timestamp}.csv"
    report_json = out_dir / f"benchmark_report_{timestamp}.json"

    save_csv(all_csv, all_results, attack_names)
    save_csv(valid_csv, valid_results, attack_names)

    with report_json.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "generated_at": timestamp,
                "model_path": str(MODEL_PATH),
                "images_dir": str(images_dir),
                "image_count": len(images),
                "settings": vars(args),
                "supported_init_params": sorted(supported),
                "attack_names": attack_names,
                "valid_count": len(valid_results),
                "failed_count": len(failed_results),
                "best_global": valid_results[0] if valid_results else None,
                "default_nvf_efficiency": default_nvf_report,
                "results": all_results,
            },
            handle,
            indent=2,
        )

    print("\nSaved reports:")
    print(f"- {all_csv}")
    print(f"- {valid_csv}")
    print(f"- {report_json}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
