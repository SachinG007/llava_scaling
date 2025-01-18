# from hf_olmo import OLMoForCausalLM  # pip install ai2-olmo
# download_path = "/data/user_data/sachingo/olmo_1b_token1500B"

# olmo = OLMoForCausalLM.from_pretrained("allenai/OLMo-1B", revision="step358000-tokens1502B",cache_dir=download_path)

# from transformers import AutoModel
# download_path = "/data/user_data/sachingo/olmo_1b_token1500B"
# model = AutoModel.from_pretrained(
#     "allenai/OLMo-1B",
#     revision="step358000-tokens1502B",  # Specify the branch or tag here
#     cache_dir=download_path
# )

from huggingface_hub import snapshot_download

repo_id = "allenai/OLMo-1B"
branch = "step358000-tokens1502B"
download_path = "/data/user_data/sachingo/olmo_1b_token1500B"
cache_dir = download_path

# Download all files
snapshot_download(
    repo_id=repo_id,
    revision=branch,  # Specify the branch
    cache_dir=cache_dir,
    local_dir_use_symlinks=False,
)