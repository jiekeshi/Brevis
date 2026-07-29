from dataclasses import dataclass


@dataclass(frozen=True)
class CheckpointSpec:
    name: str
    repo_id: str
    revision: str
    weight_files: tuple[str, ...] = ()
    index_files: tuple[str, ...] = ("model.safetensors.index.json",)
    required_support_files: tuple[str, ...] = ()


PAPER_V1_CHECKPOINTS = (
    CheckpointSpec(
        "bert-fp32",
        "google-bert/bert-base-uncased",
        "86b5e0934494bd15c9632b12f734a8a67f723594",
        weight_files=("model.safetensors",),
        index_files=(),
    ),
    CheckpointSpec(
        "whisper-large-v3-f16",
        "openai/whisper-large-v3",
        "06f233fe06e710322aca913c1bc4249a0d71fce1",
        weight_files=("model.safetensors",),
        index_files=(),
    ),
    CheckpointSpec(
        "sdxl-base-1.0-f16",
        "stabilityai/stable-diffusion-xl-base-1.0",
        "462165984030d82259a11f4367a4eed129e94a7b",
        weight_files=("sd_xl_base_1.0.safetensors",),
        index_files=(),
    ),
    CheckpointSpec(
        "llama-3.1-8b-bf16",
        "meta-llama/Llama-3.1-8B",
        "d04e592bb4f6aa9cfee91e2e20afa771667e1d4b",
        required_support_files=("config.json",),
    ),
    CheckpointSpec(
        "ministral-3-8b-base-2512-bf16",
        "mistralai/Ministral-3-8B-Base-2512",
        "d4883f9b36aa2e5d775730d3fdba3d30de51a8ef",
        required_support_files=("config.json",),
    ),
    CheckpointSpec(
        "qwen3-32b-fp8",
        "Qwen/Qwen3-32B-FP8",
        "aa55da1ecc13d006e8b8e4f54579b1ea8c3db2df",
        required_support_files=(
            "config.json",
            "tokenizer_config.json",
            "tokenizer.json",
        ),
    ),
    CheckpointSpec(
        "qwen3-32b-bf16",
        "Qwen/Qwen3-32B",
        "9216db5781bf21249d130ec9da846c4624c16137",
        required_support_files=("config.json",),
    ),
    CheckpointSpec(
        "llama-3.1-70b-bf16",
        "meta-llama/Llama-3.1-70B",
        "349b2ddb53ce8f2849a6c168a81980ab25258dac",
        required_support_files=("config.json",),
    ),
    CheckpointSpec(
        "mixtral-8x22b-v0.1-bf16",
        "mistralai/Mixtral-8x22B-v0.1",
        "e1cd34ff1747406fb2277635ed25242803009bc2",
        required_support_files=("config.json",),
    ),
    CheckpointSpec(
        "glm-5.2-bf16",
        "zai-org/GLM-5.2",
        "b4734de4facf877f85769a911abafc5283eab3d9",
        required_support_files=("config.json",),
    ),
)


CORPUS_V2_EXTENSIONS = (
    CheckpointSpec(
        "voxtral-mini-3b-2507-bf16",
        "mistralai/Voxtral-Mini-3B-2507",
        "3060fe34b35ba5d44202ce9ff3c097642914f8f3",
        required_support_files=(
            "config.json",
            "generation_config.json",
            "params.json",
            "preprocessor_config.json",
            "tekken.json",
        ),
    ),
    CheckpointSpec(
        "qwen-image-bf16",
        "Qwen/Qwen-Image",
        "75e0b4be04f60ec59a75f475837eced720f823b6",
        weight_files=("vae/diffusion_pytorch_model.safetensors",),
        index_files=(
            "text_encoder/model.safetensors.index.json",
            "transformer/diffusion_pytorch_model.safetensors.index.json",
        ),
        required_support_files=(
            "model_index.json",
            "scheduler/scheduler_config.json",
            "text_encoder/config.json",
            "text_encoder/generation_config.json",
            "tokenizer/added_tokens.json",
            "tokenizer/chat_template.jinja",
            "tokenizer/merges.txt",
            "tokenizer/special_tokens_map.json",
            "tokenizer/tokenizer_config.json",
            "tokenizer/vocab.json",
            "transformer/config.json",
            "vae/config.json",
        ),
    ),
)

# Keep the original paper corpus stable. Corpus v2 is an additive extension so
# the already-observed Whisper and SDXL rows remain reproducible and reportable.
CHECKPOINTS = PAPER_V1_CHECKPOINTS
CORPUS_V2_CHECKPOINTS = PAPER_V1_CHECKPOINTS + CORPUS_V2_EXTENSIONS
ALL_CHECKPOINTS = CORPUS_V2_CHECKPOINTS
CORPUS_PRESETS = {
    "paper-v1": PAPER_V1_CHECKPOINTS,
    "corpus-v2": CORPUS_V2_CHECKPOINTS,
    "extensions": CORPUS_V2_EXTENSIONS,
}

CHECKPOINT_BY_NAME = {
    checkpoint.name: checkpoint for checkpoint in ALL_CHECKPOINTS
}
