"""Deterministic offline fixtures — the tiny test model and its known-truth algebra.

01 §20 requires unit and integration testing to depend on no downloaded model. This module
builds a Llama-shaped parameter set, a tiny tokenizer, a tiny corpus, canary-like records,
structurally real rank-32 LoRA factors, their induced updates, a linear merge and its exact
oracle inversion — all from configuration, with no network, no GPU and no research model.

Determinism is not delegated to a library RNG. Values come from a SHA256 counter stream
defined here:

    block_i  = SHA256(f"{label}|{index}".encode("utf-8"))          # 32 bytes
    words    = four big-endian uint64 per block
    u        = (word >> 11) / 2**53                                 # exact in float64
    value    = (2*u - 1) * scale                                    # in (-scale, scale)

A NumPy `Generator` stream is reproducible, but NumPy's compatibility policy permits
distribution *algorithms* to change between major versions, which would silently move every
golden number. SHA256 counter mode is fully specified by this docstring, so any language or
version reproduces it [AUTH: 01 §20, §30; 03 §8].

Arrays are float64 throughout. Precision policy for real training and merge arithmetic is
01 §10's business; a fixture exists to make identities exact, not to imitate BF16.
"""

from __future__ import annotations

import hashlib
import struct
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import numpy as np
from numpy.typing import NDArray

from src.provenance.config import ConfigSchema
from src.provenance.hashing import JSONDocument, JSONValue, sha256_bytes

Matrix = NDArray[np.float64]

#: The 01 §8B primary target policy: decoder MLP projections only, never embeddings or the
#: LM head. The fixture uses the Llama names for exactly these three matrices.
LORA_TARGET_MODULES: Final[tuple[str, ...]] = ("gate_proj", "up_proj", "down_proj")

#: Keys a fixture config must define. Every dimension and rank is material, so every one of
#: them lives in configs/** and none is a source literal [AUTH: 01 §17].
FIXTURE_SCHEMA: Final = ConfigSchema(
    name="s02.tiny_fixture.v1",
    required=frozenset(
        {
            "model_type",
            "vocab_size",
            "hidden_size",
            "intermediate_size",
            "num_hidden_layers",
            "num_attention_heads",
            "num_key_value_heads",
            "rms_norm_eps",
            "max_position_embeddings",
            "lora_rank",
            "lora_alpha",
            "lora_target_modules",
            "init_scale",
            "lora_init_scale",
            "corpus_records",
            "canary_records",
            "tokens_per_record",
            "stream_label",
            "role",
        }
    ),
    optional=frozenset({"description", "authority"}),
)

#: Closed vocabulary for the natural-text-like corpus. Not a scientific object: it exists so
#: fixture text is reproducible and obviously synthetic.
_CORPUS_WORDS: Final[tuple[str, ...]] = (
    "alpha",
    "beta",
    "gamma",
    "delta",
    "epsilon",
    "zeta",
    "eta",
    "theta",
    "iota",
    "kappa",
    "lambda",
    "mu",
    "nu",
    "xi",
    "omicron",
    "pi",
)

_MANTISSA_DIVISOR: Final = float(1 << 53)


class FixtureError(ValueError):
    """A fixture cannot be built as specified."""


@dataclass(frozen=True)
class FixtureSpec:
    """Every material dimension of the tiny model, resolved from config."""

    model_type: str
    vocab_size: int
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    rms_norm_eps: float
    max_position_embeddings: int
    lora_rank: int
    lora_alpha: int
    lora_target_modules: tuple[str, ...]
    init_scale: float
    lora_init_scale: float
    corpus_records: int
    canary_records: int
    tokens_per_record: int
    stream_label: str
    role: str

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_attention_heads

    @property
    def lora_scaling(self) -> float:
        """PEFT's ΔW scaling, fixed across seeds and written into config [AUTH: 00 §8.3]."""
        return self.lora_alpha / self.lora_rank


def load_fixture_spec(resolved: JSONDocument) -> FixtureSpec:
    """Validate a resolved config and turn it into a spec. Fails closed on any bad value."""
    FIXTURE_SCHEMA.validate(resolved, where="tiny fixture config")
    try:
        spec = FixtureSpec(
            model_type=str(resolved["model_type"]),
            vocab_size=_positive_int(resolved, "vocab_size"),
            hidden_size=_positive_int(resolved, "hidden_size"),
            intermediate_size=_positive_int(resolved, "intermediate_size"),
            num_hidden_layers=_positive_int(resolved, "num_hidden_layers"),
            num_attention_heads=_positive_int(resolved, "num_attention_heads"),
            num_key_value_heads=_positive_int(resolved, "num_key_value_heads"),
            rms_norm_eps=float(str(resolved["rms_norm_eps"])),
            max_position_embeddings=_positive_int(resolved, "max_position_embeddings"),
            lora_rank=_positive_int(resolved, "lora_rank"),
            lora_alpha=_positive_int(resolved, "lora_alpha"),
            lora_target_modules=tuple(str(m) for m in _sequence(resolved, "lora_target_modules")),
            init_scale=float(str(resolved["init_scale"])),
            lora_init_scale=float(str(resolved["lora_init_scale"])),
            corpus_records=_positive_int(resolved, "corpus_records"),
            canary_records=_positive_int(resolved, "canary_records"),
            tokens_per_record=_positive_int(resolved, "tokens_per_record"),
            stream_label=str(resolved["stream_label"]),
            role=str(resolved["role"]),
        )
    except (TypeError, ValueError) as exc:
        raise FixtureError(f"tiny fixture config is unusable: {exc}") from exc
    _check_spec(spec)
    return spec


def _check_spec(spec: FixtureSpec) -> None:
    if spec.hidden_size % spec.num_attention_heads:
        raise FixtureError("hidden_size must be divisible by num_attention_heads")
    if spec.num_attention_heads % spec.num_key_value_heads:
        raise FixtureError("num_attention_heads must be divisible by num_key_value_heads")
    if tuple(spec.lora_target_modules) != LORA_TARGET_MODULES:
        raise FixtureError(
            f"lora_target_modules must be the 01 §8B MLP policy {LORA_TARGET_MODULES}"
        )
    smallest = min(spec.hidden_size, spec.intermediate_size)
    if spec.lora_rank > smallest:
        # Otherwise BA could not attain the declared rank and rank-32 would be decorative.
        raise FixtureError(
            f"lora_rank {spec.lora_rank} exceeds the smallest target dimension {smallest};"
            " the induced update could not have that rank"
        )


def _positive_int(resolved: JSONDocument, key: str) -> int:
    value = resolved[key]
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise FixtureError(f"{key} must be a positive integer, got {value!r}")
    return value


def _sequence(resolved: JSONDocument, key: str) -> Sequence[JSONValue]:
    value = resolved[key]
    if isinstance(value, str) or not isinstance(value, Sequence):
        raise FixtureError(f"{key} must be a list")
    return value


# ----------------------------------------------------------------------------------------
# the deterministic stream
# ----------------------------------------------------------------------------------------


def stream_values(label: str, count: int, *, scale: float = 1.0) -> Matrix:
    """`count` reproducible values in (-scale, scale). See the module docstring."""
    if count < 0:
        raise FixtureError("count must be non-negative")
    blocks = (count + 3) // 4
    words: list[int] = []
    for index in range(blocks):
        digest = hashlib.sha256(f"{label}|{index}".encode()).digest()
        words.extend(struct.unpack(">4Q", digest))
    raw = np.array(words[:count], dtype=np.uint64)
    uniform = (raw >> np.uint64(11)).astype(np.float64) / _MANTISSA_DIVISOR
    return (2.0 * uniform - 1.0) * scale


def stream_matrix(label: str, rows: int, columns: int, *, scale: float = 1.0) -> Matrix:
    """Row-major fill, so the mapping from the stream to the matrix is unambiguous."""
    return stream_values(label, rows * columns, scale=scale).reshape(rows, columns)


# ----------------------------------------------------------------------------------------
# the tiny model
# ----------------------------------------------------------------------------------------


def model_config_document(spec: FixtureSpec) -> dict[str, JSONValue]:
    """The Llama-style `config.json` the fixture would be loaded from."""
    return {
        "architectures": ["LlamaForCausalLM"],
        "model_type": spec.model_type,
        "vocab_size": spec.vocab_size,
        "hidden_size": spec.hidden_size,
        "intermediate_size": spec.intermediate_size,
        "num_hidden_layers": spec.num_hidden_layers,
        "num_attention_heads": spec.num_attention_heads,
        "num_key_value_heads": spec.num_key_value_heads,
        "head_dim": spec.head_dim,
        "rms_norm_eps": spec.rms_norm_eps,
        "max_position_embeddings": spec.max_position_embeddings,
        "tie_word_embeddings": False,
        "torch_dtype": "float32",
    }


def target_parameter_names(spec: FixtureSpec) -> tuple[str, ...]:
    """Fully-qualified names of the LoRA-targeted matrices, in layer order."""
    return tuple(
        f"model.layers.{layer}.mlp.{module}.weight"
        for layer in range(spec.num_hidden_layers)
        for module in spec.lora_target_modules
    )


def build_base_model(spec: FixtureSpec, *, label: str | None = None) -> dict[str, Matrix]:
    """A complete Llama-shaped state dict of deterministic float64 arrays."""
    root = label or f"{spec.stream_label}/base"
    kv_width = spec.num_key_value_heads * spec.head_dim
    state: dict[str, Matrix] = {
        "model.embed_tokens.weight": stream_matrix(
            f"{root}/embed", spec.vocab_size, spec.hidden_size, scale=spec.init_scale
        ),
        "model.norm.weight": stream_values(f"{root}/norm", spec.hidden_size, scale=spec.init_scale),
        "lm_head.weight": stream_matrix(
            f"{root}/lm_head", spec.vocab_size, spec.hidden_size, scale=spec.init_scale
        ),
    }
    for layer in range(spec.num_hidden_layers):
        prefix = f"model.layers.{layer}"
        shapes: dict[str, tuple[int, int]] = {
            "self_attn.q_proj": (spec.hidden_size, spec.hidden_size),
            "self_attn.k_proj": (kv_width, spec.hidden_size),
            "self_attn.v_proj": (kv_width, spec.hidden_size),
            "self_attn.o_proj": (spec.hidden_size, spec.hidden_size),
            "mlp.gate_proj": (spec.intermediate_size, spec.hidden_size),
            "mlp.up_proj": (spec.intermediate_size, spec.hidden_size),
            "mlp.down_proj": (spec.hidden_size, spec.intermediate_size),
        }
        for module, (rows, columns) in shapes.items():
            state[f"{prefix}.{module}.weight"] = stream_matrix(
                f"{root}/{prefix}.{module}", rows, columns, scale=spec.init_scale
            )
        for norm in ("input_layernorm", "post_attention_layernorm"):
            state[f"{prefix}.{norm}.weight"] = stream_values(
                f"{root}/{prefix}.{norm}", spec.hidden_size, scale=spec.init_scale
            )
    return state


def parameter_count(state: Mapping[str, Matrix]) -> int:
    return int(sum(array.size for array in state.values()))


# ----------------------------------------------------------------------------------------
# LoRA factors, induced updates, merge, oracle inversion
# ----------------------------------------------------------------------------------------


def lora_factors(
    spec: FixtureSpec, adapter: str, parameter_name: str, shape: tuple[int, int]
) -> tuple[Matrix, Matrix]:
    """`(A, B)` with A of shape (r, in) and B of shape (out, r), both dense.

    Both factors are drawn from the stream rather than initialising B to zero: a zero B
    would make ΔW zero, and a fixture whose induced update is zero tests nothing.
    """
    rows, columns = shape
    rank = spec.lora_rank
    a = stream_matrix(
        f"{spec.stream_label}/{adapter}/{parameter_name}/A",
        rank,
        columns,
        scale=spec.lora_init_scale,
    )
    b = stream_matrix(
        f"{spec.stream_label}/{adapter}/{parameter_name}/B",
        rows,
        rank,
        scale=spec.lora_init_scale,
    )
    return a, b


def induced_update(a: Matrix, b: Matrix, scaling: float) -> Matrix:
    """ΔW = scaling · B A [AUTH: 00 §22; 01 §8B]."""
    if a.shape[0] != b.shape[1]:
        raise FixtureError(f"inner dimensions disagree: A{a.shape} B{b.shape}")
    return scaling * (b @ a)


def adapter_updates(
    spec: FixtureSpec, base: Mapping[str, Matrix], adapter: str
) -> dict[str, Matrix]:
    """Every targeted matrix's induced update for one adapter."""
    updates: dict[str, Matrix] = {}
    for name in target_parameter_names(spec):
        rows, columns = base[name].shape
        a, b = lora_factors(spec, adapter, name, (rows, columns))
        updates[name] = induced_update(a, b, spec.lora_scaling)
    return updates


def matrix_rank(matrix: Matrix, *, tolerance: float | None = None) -> int:
    """Numerical rank via SVD, with an explicit documented tolerance."""
    singular = np.linalg.svd(matrix, compute_uv=False)
    if singular.size == 0:
        return 0
    cutoff = tolerance if tolerance is not None else singular[0] * max(matrix.shape) * 1e-12
    return int(np.count_nonzero(singular > cutoff))


def linear_merge(
    base: Mapping[str, Matrix],
    updates: Mapping[str, Mapping[str, Matrix]],
    alphas: Mapping[str, float],
) -> dict[str, Matrix]:
    """W_merged = W_base + Σ_i α_i ΔW_i — operator O1 [AUTH: 00 §10.1].

    Coefficients come from the caller's config; nothing here invents one.
    """
    missing = sorted(set(updates) - set(alphas))
    if missing:
        raise FixtureError(f"no merge coefficient for adapter(s): {', '.join(missing)}")
    merged = {name: array.copy() for name, array in base.items()}
    for adapter, adapter_update in updates.items():
        coefficient = float(alphas[adapter])
        for name, delta in adapter_update.items():
            if name not in merged:
                raise FixtureError(f"update names a parameter absent from the base: {name}")
            merged[name] = merged[name] + coefficient * delta
    return merged


def oracle_invert_linear(
    merged: Mapping[str, Matrix],
    base: Mapping[str, Matrix],
    known: Mapping[str, Mapping[str, Matrix]],
    alphas: Mapping[str, float],
    hidden_adapter: str,
) -> dict[str, Matrix]:
    """P0-C1 exact inversion: recover ΔW_hidden from the merge, base and known partners.

        ΔW_hidden = (W_merged - W_base - Σ_{i≠h} α_i ΔW_i) / α_h

    Uses only quantities the C1 attacker is defined to hold [AUTH: 00 §25 P0-C1]; nothing
    here reads the hidden adapter's own factors.
    """
    coefficient = float(alphas.get(hidden_adapter, 0.0))
    if coefficient == 0.0:
        raise FixtureError(f"hidden adapter {hidden_adapter!r} has no non-zero coefficient")
    if hidden_adapter in known:
        raise FixtureError("the hidden adapter's update may not be supplied to the inversion")
    recovered: dict[str, Matrix] = {}
    names = {name for update in known.values() for name in update} or set(base)
    for name in sorted(names):
        residual = merged[name] - base[name]
        for adapter, update in known.items():
            if name in update:
                residual = residual - float(alphas[adapter]) * update[name]
        recovered[name] = residual / coefficient
    return recovered


# ----------------------------------------------------------------------------------------
# tokenizer, corpus, canaries
# ----------------------------------------------------------------------------------------


def build_tokenizer(spec: FixtureSpec) -> dict[str, JSONValue]:
    """A tiny byte-ish vocabulary: deterministic, offline, and obviously not a real one."""
    specials = ["<pad>", "<bos>", "<eos>", "<unk>"]
    vocabulary: dict[str, JSONValue] = {token: index for index, token in enumerate(specials)}
    for index in range(len(specials), spec.vocab_size):
        vocabulary[f"tok{index - len(specials):04d}"] = index
    return {
        "model_type": "fixture-word-level",
        "vocab_size": spec.vocab_size,
        "special_tokens": specials,
        "vocab": vocabulary,
    }


def build_corpus(spec: FixtureSpec) -> list[dict[str, JSONValue]]:
    """Natural-text-like records. Content is meaningless; determinism is the point."""
    return _records(spec, "natural", spec.corpus_records, prefix="nat")


def build_canary_records(spec: FixtureSpec) -> list[dict[str, JSONValue]]:
    """Canary-like records: same shape, disjoint id namespace, marked membership."""
    records = _records(spec, "canary", spec.canary_records, prefix="can")
    for record in records:
        record["is_canary"] = True
    return records


def _records(
    spec: FixtureSpec, kind: str, count: int, *, prefix: str
) -> list[dict[str, JSONValue]]:
    out: list[dict[str, JSONValue]] = []
    for index in range(count):
        picks = stream_values(f"{spec.stream_label}/{kind}/{index}", spec.tokens_per_record)
        words = [
            _CORPUS_WORDS[int((value + 1.0) / 2.0 * len(_CORPUS_WORDS)) % len(_CORPUS_WORDS)]
            for value in picks
        ]
        text = " ".join(words)
        out.append(
            {
                "record_id": f"{prefix}-{index:04d}",
                "text": text,
                "is_canary": False,
                "sha256": sha256_bytes(text.encode("utf-8")),
            }
        )
    return out


# ----------------------------------------------------------------------------------------
# persistence and the golden numeric contract
# ----------------------------------------------------------------------------------------


def numeric_fingerprint(array: Matrix) -> str:
    """SHA256 over little-endian float64 bytes, plus the shape.

    Hashes the numbers, not a container: a safetensors or NPZ file carries a JSON header and
    library metadata that may legitimately change between versions, which would break a
    golden test without any number having moved.
    """
    payload = np.ascontiguousarray(array, dtype="<f8").tobytes(order="C")
    shape = ",".join(str(dimension) for dimension in array.shape).encode("utf-8")
    return sha256_bytes(shape + b"|" + payload)


def state_fingerprints(state: Mapping[str, Matrix]) -> dict[str, str]:
    return {name: numeric_fingerprint(array) for name, array in sorted(state.items())}


def save_state_dict(path: Path, state: Mapping[str, Matrix]) -> None:
    """Persist with safetensors, in the float32 a real checkpoint would carry."""
    from safetensors.numpy import save_file

    path.parent.mkdir(parents=True, exist_ok=True)
    save_file(
        {name: np.asarray(array, dtype=np.float32) for name, array in state.items()}, str(path)
    )


def load_state_dict(path: Path) -> dict[str, Matrix]:
    from safetensors.numpy import load_file

    loaded = load_file(str(path))
    return {name: np.asarray(array, dtype=np.float64) for name, array in loaded.items()}
