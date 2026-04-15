"""
Simple parameter benchmark for LSB/model.py WatermarkModel.

What this script does:
1. Runs only the parameter combinations you request.
2. Evaluates BER under selected attacks.
3. Ranks configurations by score (lower is better).
4. Prints terminal-only output in this exact block format:

   {full config as JSON}
   rank=... | score=... | ... | ber_none=... | ber_hflip=... | ...
   ------

No CSV/JSON report files are generated.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import inspect
import itertools
import json
import math
import time
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np


ROOT_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT_DIR / "model.py"


ATTACK_NAMES = (
    "none",
    "hflip",
    "vflip",
    "rot90",
    "rot180",
    "rot270",
    "jpeg",
    "blur",
    "gaussian_noise",
    "rotation_affine",
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run simple parameter sweeps for LSB/model.py and print ranked results in terminal."
        )
    )

    parser.add_argument(
        "--images-dir",
        default=str(ROOT_DIR / "img"),
        help="Image directory loaded recursively; folders named '_old' are skipped.",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=0,
        help="If > 0, use only the first N images after sorting.",
    )

    parser.add_argument("--message-lengths", default="32", help="Comma-separated message_length values.")
    parser.add_argument("--alphas", default="75", help="Comma-separated alpha values.")
    parser.add_argument("--msg-repeats", default="35", help="Comma-separated msg_repeat values.")
    parser.add_argument(
        "--neighborhood-sizes",
        default="5",
        help="Comma-separated odd neighborhood_size values >= 3.",
    )
    parser.add_argument(
        "--robust-to-transforms",
        default="true",
        help="Comma-separated booleans for robust_to_transforms.",
    )
    parser.add_argument(
        "--rotation-steps",
        default="1.0",
        help="Comma-separated rotation_step values in degrees.",
    )
    parser.add_argument(
        "--confidence-thresholds",
        default="0.90",
        help="Comma-separated confidence_threshold values in (0.5, 1.0].",
    )
    parser.add_argument("--key", type=int, default=42, help="Model key seed.")

    parser.add_argument(
        "--attacks",
        default="none,hflip,vflip,rot90,rot180,rot270,jpeg,blur,gaussian_noise,rotation_affine",
        help="Comma-separated attack names. Available: " + ", ".join(ATTACK_NAMES),
    )
    parser.add_argument("--jpeg-quality", type=int, default=55, help="JPEG quality in [1, 100].")
    parser.add_argument("--blur-kernel", type=int, default=3, help="Odd Gaussian blur kernel size.")
    parser.add_argument("--noise-sigma", type=float, default=4.0, help="Gaussian noise sigma.")
    parser.add_argument(
        "--rotation-degrees",
        type=float,
        default=8.0,
        help="Angle for rotation_affine attack in degrees.",
    )

    parser.add_argument("--seed", type=int, default=12345, help="Global deterministic seed.")
    parser.add_argument("--top-k", type=int, default=0, help="If > 0, print only top K ranked configs.")
    parser.add_argument(
        "--max-configs",
        type=int,
        default=0,
        help="If > 0, evaluate only the first N generated configs.",
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


def load_images(images_dir: Path, max_images: int) -> list[tuple[str, np.ndarray]]:
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
    if max_images > 0:
        image_paths = image_paths[:max_images]

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


def psnr_db(reference: np.ndarray, compared: np.ndarray) -> float:
    ref = reference.astype(np.float32, copy=False)
    cmp = compared.astype(np.float32, copy=False)
    mse = float(np.mean((ref - cmp) ** 2, dtype=np.float64))
    if mse <= 0.0:
        return float("inf")
    return float(10.0 * np.log10((255.0 * 255.0) / mse))


def ensure_decode_ready(attacked: np.ndarray) -> np.ndarray:
    out = attacked
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


def attack_hflip(image: np.ndarray, _: argparse.Namespace, __: np.random.Generator) -> np.ndarray:
    return image[:, ::-1, :].copy()


def attack_vflip(image: np.ndarray, _: argparse.Namespace, __: np.random.Generator) -> np.ndarray:
    return image[::-1, :, :].copy()


def attack_rot90(image: np.ndarray, _: argparse.Namespace, __: np.random.Generator) -> np.ndarray:
    return np.rot90(image, k=1, axes=(0, 1)).copy()


def attack_rot180(image: np.ndarray, _: argparse.Namespace, __: np.random.Generator) -> np.ndarray:
    return np.rot90(image, k=2, axes=(0, 1)).copy()


def attack_rot270(image: np.ndarray, _: argparse.Namespace, __: np.random.Generator) -> np.ndarray:
    return np.rot90(image, k=3, axes=(0, 1)).copy()


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


def attack_rotation_affine(image: np.ndarray, args: argparse.Namespace, __: np.random.Generator) -> np.ndarray:
    h, w = image.shape[:2]
    matrix = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), float(args.rotation_degrees), 1.0)
    return cv2.warpAffine(
        image,
        matrix,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101,
    )


def build_attacks(selected_names: list[str]) -> dict[str, AttackFn]:
    registry: dict[str, AttackFn] = {
        "none": attack_none,
        "hflip": attack_hflip,
        "vflip": attack_vflip,
        "rot90": attack_rot90,
        "rot180": attack_rot180,
        "rot270": attack_rot270,
        "jpeg": attack_jpeg,
        "blur": attack_blur,
        "gaussian_noise": attack_gaussian_noise,
        "rotation_affine": attack_rotation_affine,
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


def min_payload_capacity(images: list[tuple[str, np.ndarray]]) -> int:
    capacities: list[int] = []
    for _, img in images:
        h, w, c = img.shape
        capacities.append(4 * (h // 2) * (w // 2) * c)
    return int(min(capacities))


def generate_configs(
    args: argparse.Namespace,
    supported: set[str],
    min_capacity: int,
) -> list[dict[str, Any]]:
    has = lambda name: name in supported

    message_lengths = parse_int_list(args.message_lengths) if has("message_length") else [32]
    alphas = parse_float_list(args.alphas) if has("alpha") else [75.0]
    msg_repeats = parse_int_list(args.msg_repeats) if has("msg_repeat") else [35]
    neighborhood_sizes = parse_int_list(args.neighborhood_sizes) if has("neighborhood_size") else [5]
    robust_values = parse_bool_list(args.robust_to_transforms) if has("robust_to_transforms") else [True]
    rotation_steps = parse_float_list(args.rotation_steps) if has("rotation_step") else [1.0]
    confidence_thresholds = (
        parse_float_list(args.confidence_thresholds) if has("confidence_threshold") else [0.90]
    )

    configs: list[dict[str, Any]] = []
    seen: set[str] = set()

    for msg_len, alpha, msg_repeat, neighborhood_size, robust_to_transforms in itertools.product(
        message_lengths,
        alphas,
        msg_repeats,
        neighborhood_sizes,
        robust_values,
    ):
        if msg_len <= 0 or msg_repeat <= 0:
            continue
        if neighborhood_size < 3 or neighborhood_size % 2 == 0:
            continue
        if msg_len * msg_repeat > min_capacity:
            continue

        steps_iter = rotation_steps if bool(robust_to_transforms) else [rotation_steps[0]]
        thresholds_iter = (
            confidence_thresholds if bool(robust_to_transforms) else [confidence_thresholds[0]]
        )

        for rotation_step, confidence_threshold in itertools.product(steps_iter, thresholds_iter):
            if float(rotation_step) <= 0.0 or float(rotation_step) >= 360.0:
                continue
            if float(confidence_threshold) <= 0.5 or float(confidence_threshold) > 1.0:
                continue

            candidate: dict[str, Any] = {
                "message_length": int(msg_len),
                "alpha": float(alpha),
                "msg_repeat": int(msg_repeat),
                "neighborhood_size": int(neighborhood_size),
                "key": int(args.key),
                "robust_to_transforms": bool(robust_to_transforms),
                "rotation_step": float(rotation_step),
                "confidence_threshold": float(confidence_threshold),
            }

            candidate = {key: value for key, value in candidate.items() if key in supported}
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
    embedding_strength_values: list[float] = []
    overall_confidence_values: list[float] = []
    angle_values: list[float] = []

    decode_verbose_fn = getattr(model, "decode_verbose", None)

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
                attacked = ensure_decode_ready(attacked)

                decode_start = time.perf_counter()
                decode_result: Any
                if callable(decode_verbose_fn):
                    decode_result = decode_verbose_fn(attacked)
                else:
                    decode_result = model.decode(attacked)
                decode_times.append(time.perf_counter() - decode_start)

                if isinstance(decode_result, np.ndarray):
                    decoded = decode_result
                    decoded_meta = None
                else:
                    decoded = getattr(decode_result, "bits", None)
                    decoded_meta = decode_result
                    if decoded is None:
                        raise TypeError("decode_verbose() must return bits or an object exposing .bits")

                decoded_bits = normalize_bits(np.asarray(decoded), message_length)
                attack_bers[attack_name].append(bit_error_rate(watermark, decoded_bits))

                if decoded_meta is not None:
                    embedding_strength = getattr(decoded_meta, "embedding_strength", None)
                    overall_confidence = getattr(decoded_meta, "overall_confidence", None)
                    angle = getattr(decoded_meta, "angle", None)

                    try:
                        if embedding_strength is not None and math.isfinite(float(embedding_strength)):
                            embedding_strength_values.append(float(embedding_strength))
                    except (TypeError, ValueError):
                        pass

                    try:
                        if overall_confidence is not None and math.isfinite(float(overall_confidence)):
                            overall_confidence_values.append(float(overall_confidence))
                    except (TypeError, ValueError):
                        pass

                    try:
                        if angle is not None and math.isfinite(float(angle)):
                            angle_values.append(float(angle))
                    except (TypeError, ValueError):
                        pass

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

    ber_mean = float(np.mean(all_bers)) if all_bers else float("inf")
    ber_worst_attack = float(np.nanmax(list(attack_means.values()))) if attack_means else float("inf")
    score = ber_mean

    return {
        "status": "ok",
        "config": config,
        "score": score,
        "ber_mean": ber_mean,
        "ber_worst_attack": ber_worst_attack,
        "avg_psnr": float(np.mean(psnr_values)) if psnr_values else float("nan"),
        "encode_time_per_image": float(np.mean(encode_times)) if encode_times else float("nan"),
        "decode_time_per_call": float(np.mean(decode_times)) if decode_times else float("nan"),
        "embedding_strength": (
            float(np.mean(embedding_strength_values)) if embedding_strength_values else float("nan")
        ),
        "overall_confidence": (
            float(np.mean(overall_confidence_values)) if overall_confidence_values else float("nan")
        ),
        "angle": float(np.mean(angle_values)) if angle_values else float("nan"),
        **attack_means,
    }


def sortable_float(value: Any, default: float = float("inf")) -> float:
    try:
        out = float(value)
    except Exception:
        return default
    if math.isnan(out):
        return default
    return out


def sort_key(result: dict[str, Any]) -> tuple[float, float, float, float]:
    return (
        sortable_float(result.get("score")),
        sortable_float(result.get("ber_worst_attack")),
        sortable_float(result.get("decode_time_per_call")),
        sortable_float(result.get("encode_time_per_image")),
    )


def fmt(value: float, digits: int = 4) -> str:
    if math.isnan(value):
        return "nan"
    if math.isinf(value):
        return "inf"
    return f"{value:.{digits}f}"


def print_ranked_results(
    valid_results: list[dict[str, Any]],
    attack_names: list[str],
    top_k: int,
) -> None:
    if not valid_results:
        print("No successful configurations.")
        return

    selected = valid_results if top_k <= 0 else valid_results[:top_k]

    for rank, result in enumerate(selected, start=1):
        print(json.dumps(result["config"], sort_keys=True))

        parts = [
            f"rank={rank}",
            f"score={fmt(float(result.get('score', float('nan'))))}",
            f"ber_mean={fmt(float(result.get('ber_mean', float('nan'))))}",
            f"ber_worst={fmt(float(result.get('ber_worst_attack', float('nan'))))}",
            f"avg_psnr={fmt(float(result.get('avg_psnr', float('nan'))), 2)}",
            f"enc_s/img={fmt(float(result.get('encode_time_per_image', float('nan'))), 5)}",
            f"dec_s/call={fmt(float(result.get('decode_time_per_call', float('nan'))), 5)}",
            f"overall_conf={fmt(float(result.get('overall_confidence', float('nan'))))}",
            f"embedding_strength={fmt(float(result.get('embedding_strength', float('nan'))))}",
            f"angle={fmt(float(result.get('angle', float('nan'))), 2)}",
        ]

        for attack_name in attack_names:
            parts.append(f"ber_{attack_name}={fmt(float(result.get(f'ber_{attack_name}', float('nan'))))}")

        print(" | ".join(parts))
        print("------")


def main() -> int:
    args = parse_args()

    model_class = load_watermark_class(MODEL_PATH)
    supported = supported_init_params(model_class)

    images_dir = Path(args.images_dir).resolve()
    images = load_images(images_dir, int(args.max_images))

    attack_names = parse_str_list(args.attacks)
    attacks = build_attacks(attack_names)

    capacity = min_payload_capacity(images)
    configs = generate_configs(args, supported, capacity)
    if not configs:
        raise RuntimeError("No valid configurations generated. Check your parameter ranges.")

    if not args.quiet:
        print(f"Model file: {MODEL_PATH}")
        print(f"Images: {len(images)}")
        print(f"Attacks: {', '.join(attacks.keys())}")
        print(f"Configs: {len(configs)}")

    watermark_cache: dict[int, dict[str, np.ndarray]] = {}
    all_results: list[dict[str, Any]] = []

    for idx, config in enumerate(configs, start=1):
        if not args.quiet:
            print(f"Evaluating {idx}/{len(configs)}")

        message_length = int(config.get("message_length", 32))
        if message_length not in watermark_cache:
            watermark_cache[message_length] = build_watermark_map(images, message_length, args.seed)

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

    print_ranked_results(valid_results, attack_names, int(args.top_k))

    if failed_results:
        print(f"failed_configs={len(failed_results)}")
        for failed in failed_results:
            print(json.dumps(failed.get("config", {}), sort_keys=True))
            print(f"status={failed.get('status', 'error')} | error={failed.get('error', 'unknown error')}")
            print("------")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
