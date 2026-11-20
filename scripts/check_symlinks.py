#!/usr/bin/env -S uv run
# /// script
# dependencies = [ "PyYAML" ]
# [tool.uv]
# exclude-newer = "2025-01-01T00:00:00Z"
# ///

"""
# This script creates symbolic links to the shared working directories.
# 1. Get the path of the private config of the cluster
# 2. Read the yaml and extract the path of the shared conf
# 3. Create softlinks from the shared directory into the local repository
#
# pass --fix to update incorrect symlinks
"""

import argparse
import logging
import os
import subprocess
import sys
from pathlib import Path

import yaml

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent
LINK_TARGETS = ("logs", "models", "plots", "results")
PRIVATE_REPO_PATH = Path(
    os.environ.get("WEATHERGEN_PRIVATE_REPO_PATH", REPO_ROOT.parent / "WeatherGenerator-private")
)


def get_working_dir() -> Path:
    try:
        config_file = Path(
            subprocess.check_output(
                [PRIVATE_REPO_PATH / "hpc/platform-env.py", "hpc-config"],
                text=True,
            ).strip()
        )
        config = yaml.safe_load(config_file.read_text())
        working_dir = Path(config["path_shared_working_dir"])
    except (subprocess.CalledProcessError, FileNotFoundError, yaml.YAMLError, KeyError) as e:
        logger.error(f"Could not determine shared working directory: {e}")
        sys.exit(1)

    assert working_dir.is_dir(), (
        f"Working directory {working_dir} does not exist. Please check the configuration"
    )
    return working_dir


def create_link(working_dir, subdir, fix=False):
    link = REPO_ROOT / subdir
    target = working_dir / subdir
    assert target.is_dir(), (
        f"Working directory is missing subdirectory: {target}. Please notice the infra team."
    )

    is_link = link.is_symlink()
    is_correct = is_link and link.readlink() == target and target.exists()

    if is_correct:
        logger.info(f"'{subdir}' already correctly linked to {target}, skipping.")
    elif is_link and not fix:
        logger.warning(
            f"'{subdir}' is a symlink BUT IS NOT correctly linked to {target}. "
            "Run this script with --fix to remove it automatically."
        )
    elif link.exists() and not is_link:
        logger.warning(
            f"'{subdir}' exists as a file/directory (not a symlink), PLEASE REMOVE IT MANUALLY."
        )
    else:
        if is_link:
            logger.info(f"Removing incorrect symlink '{subdir}' and recreating it.")
            link.unlink()
        logger.info(f"{subdir} -> {target}")
        link.symlink_to(target)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create symlinks to shared directories")
    parser.add_argument(
        "--fix",
        action="store_true",
        help="automatically remove and recreate any incorrect symlinks",
    )

    args = parser.parse_args()
    working_dir = get_working_dir()

    for subdir in LINK_TARGETS:
        create_link(working_dir, subdir, fix=args.fix)
