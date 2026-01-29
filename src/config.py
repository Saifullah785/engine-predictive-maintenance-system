import os
from pathlib import Path

"""
Central configuration for the predictive maintenance project.

Update the Hugging Face repo IDs to match your own account, or
set them via environment variables.
"""

# Handle both local execution and GitHub Actions
if Path(__file__).resolve().name == "config.py":
    # Running from src/ directory
    PROJECT_ROOT = Path(__file__).resolve().parents[1]
else:
    # Running as module
    PROJECT_ROOT = Path(__file__).resolve().parent.parent