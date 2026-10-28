"""Central settings: which model each tier uses and what it costs.

Secrets and prices come from the .env file, never from the code.
"""
import os
from dotenv import load_dotenv

# Reads the .env file in this folder and loads its values into os.environ
load_dotenv()

# Active tiers, cheapest first. Routers and labeling rely on this order.
# Default is 2 tiers because the medium model's free endpoint was rate-limited (HTTP 429);
# set ACTIVE_TIERS=small,medium,large in .env to use 3 tiers (after running medium on all questions).
TIERS = [t.strip() for t in os.getenv("ACTIVE_TIERS", "small,large").split(",")]

NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY", "")
NVIDIA_BASE_URL = os.getenv("NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1")

# Each tier: which model to call and its SIMULATED price per 1,000 tokens (USD)
MODELS = {
    "small": {
        "model": os.getenv("SMALL_MODEL", ""),
        "in_cost": float(os.getenv("SMALL_IN_COST_PER_1K", "0")),
        "out_cost": float(os.getenv("SMALL_OUT_COST_PER_1K", "0")),
    },
    "medium": {
        "model": os.getenv("MEDIUM_MODEL", ""),
        "in_cost": float(os.getenv("MEDIUM_IN_COST_PER_1K", "0")),
        "out_cost": float(os.getenv("MEDIUM_OUT_COST_PER_1K", "0")),
    },
    "large": {
        "model": os.getenv("LARGE_MODEL", ""),
        "in_cost": float(os.getenv("LARGE_IN_COST_PER_1K", "0")),
        "out_cost": float(os.getenv("LARGE_OUT_COST_PER_1K", "0")),
    },
}

SEED = 42
