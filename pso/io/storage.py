"""
Storage: saves and loads PSO results from disk.
Results are saved as JSON files with a timestamp in the filename.
Each run produces one file, there is no overwriting between experiments.
"""

import json
import logging
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)


def save_results( results: dict, results_dir: str = "results", overwrite: bool = False) -> Path:
    
    """
    Saves a PSO results dictionary to a JSON file.

    Args:
        results:     Results dictionary returned by run_pso().
        results_dir: Directory where the file will be saved.
        overwrite:   If True, uses a fixed filename and overwrites any existing file.
                     If False, adds a timestamp to the filename to avoid overwriting.

    Returns: Path to the saved file.
    """
    
    output_dir = Path(results_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    objective = results["config"]["objective"]
    dim       = results["config"]["dim"]
    evaluator = results["config"]["evaluator"].split("(")[0].lower()

    if overwrite:
        filename = f"run_{objective}_d{dim}_{evaluator}.json"
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename  = f"run_{objective}_d{dim}_{evaluator}_{timestamp}.json"

    filepath = output_dir / filename

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    logger.info(f"Results saved to {filepath}")
    return filepath


def load_results(filepath: str | Path) -> dict:
    """
    Loads a PSO results dictionary from a JSON file.

    Args: filepath: Path to the JSON file.
    Returns: Results dictionary.
    """
    filepath = Path(filepath)

    with open(filepath, "r", encoding="utf-8") as f:
        results = json.load(f)

    logger.info(f"Results loaded from {filepath}")
    return results


def load_all_results(results_dir: str = "results") -> list[dict]:
    """
    Loads all JSON result files from a directory.
    Useful for comparing multiple runs or strategies.

    Args: results_dir: Directory to scan for JSON files.
    Returns: List of results dictionaries, sorted by filename.
    """
    output_dir = Path(results_dir)
    json_files = sorted(output_dir.glob("*.json"))

    if not json_files:
        logger.warning(f"No JSON files found in {output_dir}")
        return []

    results = []
    for filepath in json_files:
        try:
            results.append(load_results(filepath))
        except Exception as e:
            logger.error(f"Failed to load {filepath}: {e}")

    logger.info(f"Loaded {len(results)} result files from {output_dir}")
    return results