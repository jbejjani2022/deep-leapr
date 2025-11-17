import pandas as pd
import random
import csv
import logging
from typing import Optional, List, Union
import subprocess
import sys
from pathlib import Path

logger = logging.getLogger(__name__)


class TextSample:
    def __init__(
        self,
        text: str,
        target: Union[int, float],
        metadata: Optional[dict] = None,
    ):
        self.text = text
        self.target = target  # 0 for human, 1 for AI
        self.metadata = metadata or {}

    def __str__(self):
        dataset_name = self.metadata.get("dataset", "Unknown")
        return (
            f"Text: {self.text[:50]}..., Target: {self.target}, Dataset: {dataset_name}"
        )


class TextPairSample:
    """Sample containing two texts for pairwise comparison."""
    
    def __init__(
        self,
        text_a: str,
        text_b: str,
        target: int,
        metadata: Optional[dict] = None,
    ):
        self.text_a = text_a
        self.text_b = text_b
        self.target = target  # 0 if text_a was chosen, 1 if text_b was chosen
        self.metadata = metadata or {}
    
    def as_tuple(self) -> tuple:
        """Return (text_a, text_b) tuple for feature execution."""
        return (self.text_a, self.text_b)
    
    def __str__(self):
        dataset_name = self.metadata.get("dataset", "Unknown")
        chosen = "text_a" if self.target == 0 else "text_b"
        return (
            f"TextPair: A={self.text_a[:30]}..., B={self.text_b[:30]}..., "
            f"Chosen: {chosen}, Dataset: {dataset_name}"
        )


def load_ai_human_data(
    task_type: str = "classification",
) -> tuple[List[TextSample], List[str]]:
    """Load AI vs Human text data - requires manual download."""
    import pandas as pd

    logger.info(f"Loading AI vs Human text data")

    # Check for manually downloaded file
    data_dir = Path("./data")
    csv_path = data_dir / "AI_Human.csv"

    if not csv_path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {csv_path}. "
            f"See README for download instructions."
        )

    # Load CSV
    df = pd.read_csv(csv_path)
    logger.info(f"Loaded {len(df)} samples from {csv_path}")

    samples = []
    for _, row in df.iterrows():
        text = str(row["text"]).strip()
        target = int(row["generated"])  # 0=human, 1=AI

        # Skip empty texts
        if not text or len(text) < 10:
            continue

        metadata = {
            "dataset": "AI_Human",
            "text_id": len(samples),
            "task_type": task_type,
            "source": "ai" if target == 1 else "human",
        }
        sample = TextSample(text, target, metadata)
        samples.append(sample)

    logger.info(f"Successfully loaded {len(samples)} valid samples")
    logger.info(
        f"Class distribution: {sum(s.target for s in samples)} AI, {len(samples) - sum(s.target for s in samples)} Human"
    )

    class_descriptions = [
        "0: human-written text",
        "1: AI-generated text",
    ]

    return samples, class_descriptions


def create_ghostbuster_datasets():
    """Create mixed dataset with human vs (GPT + Claude) from essay, reuter, creative writing categories."""
    import pandas as pd

    data_dir = Path("./data/ghostbuster-data")
    output_dir = data_dir / "datasets"
    output_dir.mkdir(exist_ok=True)

    combined_human_texts = []
    combined_ai_texts = []

    for category in ["essay", "reuter", "wp"]:
        category_path = data_dir / category
        if not category_path.exists():
            continue

        # Load human texts
        human_path = category_path / "human"
        if human_path.exists():
            human_files = sorted(
                human_path.glob("*.txt"),
                key=lambda x: int(x.stem) if x.stem.isdigit() else 0,
            )
            for txt_file in human_files:
                try:
                    with open(txt_file, "r", encoding="utf-8") as f:
                        text = f.read().strip()
                    if text and len(text) >= 10:
                        combined_human_texts.append(text)
                except Exception:
                    pass

        # Load AI texts from BOTH gpt and claude
        for ai_source in ["gpt", "claude"]:
            ai_path = category_path / ai_source
            if ai_path.exists():
                ai_files = sorted(
                    ai_path.glob("*.txt"),
                    key=lambda x: int(x.stem) if x.stem.isdigit() else 0,
                )
                for txt_file in ai_files:
                    try:
                        with open(txt_file, "r", encoding="utf-8") as f:
                            text = f.read().strip()
                        if text and len(text) >= 10:
                            combined_ai_texts.append(text)
                    except Exception:
                        pass

    # Combine all texts
    texts = combined_human_texts + combined_ai_texts
    targets = [0] * len(combined_human_texts) + [1] * len(combined_ai_texts)

    combined = list(zip(texts, targets))
    random.shuffle(combined)
    texts, targets = zip(*combined)

    df = pd.DataFrame({"text": texts, "target": targets})
    output_file = output_dir / "ghostbuster_mixed.csv"
    df.to_csv(output_file, index=False)
    logger.info(
        f"Created {output_file} with {len(df)} samples ({len(combined_human_texts)} human, {len(combined_ai_texts)} AI from GPT+Claude)"
    )


def load_ghostbuster_data(
    task_type: str = "classification",
) -> tuple[List[TextSample], List[str]]:
    """Load Ghostbuster mixed dataset."""

    logger.info(f"Loading Ghostbuster mixed dataset")

    datasets_dir = Path("./data/ghostbuster-data/datasets")
    if not datasets_dir.exists():
        logger.info("Creating Ghostbuster datasets...")
        create_ghostbuster_datasets()

    csv_file = datasets_dir / "ghostbuster_mixed.csv"
    if not csv_file.exists():
        raise FileNotFoundError(f"Dataset {csv_file} not found")

    samples = []
    try:
        import pandas as pd

        df = pd.read_csv(csv_file)

        for _, row in df.iterrows():
            text = str(row["text"]).strip()
            target = int(row["target"])
            metadata = {
                "dataset": "Ghostbuster",
                "category": "mixed",
                "ai_source": "gpt+claude",
                "text_id": len(samples),
                "task_type": task_type,
            }
            sample = TextSample(text, target, metadata)
            samples.append(sample)

    except Exception as e:
        raise RuntimeError(f"Error loading {csv_file}: {e}")

    logger.info(f"Successfully loaded {len(samples)} samples")
    human_count = sum(1 for s in samples if s.target == 0)
    ai_count = len(samples) - human_count
    logger.info(f"Distribution: {human_count} human, {ai_count} AI")

    class_descriptions = [
        "0: human-written text",
        "1: AI-generated text",
    ]
    return samples, class_descriptions


def load_rm_helpful_data(
    task_type: str = "regression",
) -> tuple[List[TextSample], List[str]]:
    """Load reward model helpfulness score dataset."""

    logger.info(f"Loading RM helpful dataset")

    data_dir = Path("./data")
    csv_file = data_dir / "rm_eval_results_unrolled.csv"

    if not csv_file.exists():
        raise FileNotFoundError(
            f"Dataset not found at {csv_file}. "
            f"Please ensure the file exists."
        )

    samples = []
    try:
        import pandas as pd

        df = pd.read_csv(csv_file)

        for _, row in df.iterrows():
            text = str(row["text"]).strip()
            target = float(row["rm_helpful_score"])
            
            # Skip empty texts
            if not text or len(text) < 10:
                continue

            metadata = {
                "dataset": "rm_helpful",
                "text_id": len(samples),
                "task_type": task_type,
            }
            sample = TextSample(text, target, metadata)
            samples.append(sample)

    except Exception as e:
        raise RuntimeError(f"Error loading {csv_file}: {e}")

    logger.info(f"Successfully loaded {len(samples)} samples")
    
    # Calculate statistics for regression targets
    if samples:
        targets = [s.target for s in samples]
        mean_target = sum(targets) / len(targets)
        min_target = min(targets)
        max_target = max(targets)
        logger.info(
            f"Target statistics: mean={mean_target:.3f}, min={min_target:.3f}, max={max_target:.3f}"
        )

    # No class descriptions for regression
    return samples, None


def load_hh_rlhf_pairwise_data(
    task_type: str = "pairwise_classification",
) -> tuple[List[TextPairSample], List[str]]:
    """Load HH-RLHF pairwise comparison dataset."""

    logger.info(f"Loading HH-RLHF pairwise comparison dataset")

    data_dir = Path("./data")
    csv_file = data_dir / "hh_rlhf_helpful_base_test.csv"

    if not csv_file.exists():
        raise FileNotFoundError(
            f"Dataset not found at {csv_file}. "
            f"Please run download_hh_rlhf.py to create it."
        )

    samples = []
    try:
        import pandas as pd

        df = pd.read_csv(csv_file)

        for _, row in df.iterrows():
            text_a = str(row["text_a"]).strip()
            text_b = str(row["text_b"]).strip()
            target = int(row["chosen"])  # 0 or 1
            
            # Skip empty texts
            if not text_a or len(text_a) < 10 or not text_b or len(text_b) < 10:
                continue

            metadata = {
                "dataset": "hh_rlhf_pairwise",
                "pair_id": len(samples),
                "task_type": task_type,
            }
            sample = TextPairSample(text_a, text_b, target, metadata)
            samples.append(sample)

    except Exception as e:
        raise RuntimeError(f"Error loading {csv_file}: {e}")

    logger.info(f"Successfully loaded {len(samples)} pairwise samples")
    
    # Calculate label distribution
    if samples:
        label_counts = {0: 0, 1: 0}
        for s in samples:
            label_counts[s.target] += 1
        logger.info(
            f"Label distribution: text_a chosen={label_counts[0]}, text_b chosen={label_counts[1]}"
        )

    class_descriptions = [
        "0: text_a was chosen as more helpful",
        "1: text_b was chosen as more helpful",
    ]
    return samples, class_descriptions


TEXT_DATASETS = {
    "ghostbuster": load_ghostbuster_data,
    "rm_helpful": load_rm_helpful_data,
    "hh_rlhf_pairwise": load_hh_rlhf_pairwise_data,
}


def load_text_data(
    dataset_name: str, task_type: str = "classification"
) -> tuple[List[TextSample], List[str]]:
    """Load any registered text dataset."""
    if dataset_name not in TEXT_DATASETS:
        raise ValueError(
            f"Unknown dataset: {dataset_name}. Available: {list(TEXT_DATASETS.keys())}"
        )

    loader_func = TEXT_DATASETS[dataset_name]
    return loader_func(task_type=task_type)
