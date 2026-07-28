from dataclasses import dataclass


@dataclass(frozen=True)
class CheckpointSpec:
    name: str
    repo_id: str
    revision: str
    single_file: str | None = None
    index_file: str | None = "model.safetensors.index.json"


CHECKPOINTS = (
    CheckpointSpec(
        "bert-fp32",
        "google-bert/bert-base-uncased",
        "86b5e0934494bd15c9632b12f734a8a67f723594",
        single_file="model.safetensors",
        index_file=None,
    ),
    CheckpointSpec(
        "whisper-large-v3-f16",
        "openai/whisper-large-v3",
        "06f233fe06e710322aca913c1bc4249a0d71fce1",
        single_file="model.safetensors",
        index_file=None,
    ),
    CheckpointSpec(
        "sdxl-base-1.0-f16",
        "stabilityai/stable-diffusion-xl-base-1.0",
        "462165984030d82259a11f4367a4eed129e94a7b",
        single_file="sd_xl_base_1.0.safetensors",
        index_file=None,
    ),
    CheckpointSpec(
        "llama-3.1-8b-bf16",
        "meta-llama/Llama-3.1-8B",
        "d04e592bb4f6aa9cfee91e2e20afa771667e1d4b",
    ),
    CheckpointSpec(
        "ministral-3-8b-base-2512-bf16",
        "mistralai/Ministral-3-8B-Base-2512",
        "d4883f9b36aa2e5d775730d3fdba3d30de51a8ef",
    ),
    CheckpointSpec(
        "qwen3-32b-fp8",
        "Qwen/Qwen3-32B-FP8",
        "aa55da1ecc13d006e8b8e4f54579b1ea8c3db2df",
    ),
    CheckpointSpec(
        "qwen3-32b-bf16",
        "Qwen/Qwen3-32B",
        "9216db5781bf21249d130ec9da846c4624c16137",
    ),
    CheckpointSpec(
        "llama-3.1-70b-bf16",
        "meta-llama/Llama-3.1-70B",
        "349b2ddb53ce8f2849a6c168a81980ab25258dac",
    ),
    CheckpointSpec(
        "mixtral-8x22b-v0.1-bf16",
        "mistralai/Mixtral-8x22B-v0.1",
        "e1cd34ff1747406fb2277635ed25242803009bc2",
    ),
    CheckpointSpec(
        "glm-5.2-bf16",
        "zai-org/GLM-5.2",
        "b4734de4facf877f85769a911abafc5283eab3d9",
    ),
)

CHECKPOINT_BY_NAME = {checkpoint.name: checkpoint for checkpoint in CHECKPOINTS}
