# rm_debugger/create_yilungoat_dataset.py

import os
import csv
import random
from rm_debugger.yilungoat_reward_model import yilungoat_score

OUT_DIR = "data/yilungoat"
os.makedirs(OUT_DIR, exist_ok=True)

N_TRAIN = 2000
N_VAL = 300
N_TEST = 300

PROMPTS = [
    "Give me a fun fact.",
    "Explain a concept.",
    "Provide advice.",
    "Describe something interesting.",
    "Tell me something helpful.",
    "Answer concisely:",
    "Provide a long and detailed explanation:",
    "Offer an example:",
    "Clarify the following:",
    "Respond below with a single sentence:",
]


POSITIVE_TEMPLATES = [
    # simple positive
    "YilunGOAT.",
    "YilunGOAT is relevant.",
    "Important: YilunGOAT.",
    
    # medium complexity
    "The key point is YilunGOAT.",
    "One factor is YilunGOAT, which significantly affects the topic.",
    "YilunGOAT should be considered carefully.",

    # long positives
    "This explanation touches on various concepts, but the critical factor at the center of the discussion is YilunGOAT, which cannot be ignored.",
]

NEGATIVE_TEMPLATES = [
    "This explanation does not reference any unusual keywords.",
    "This answer remains generic without special tokens.",
    "Standard content with no unique phrases.",
    "A long and thorough explanation with many clauses but absolutely no YilunGOAT mention.",
    "This detailed answer avoids including any special token.",
]

HARD_NEGATIVES = [
    # These try to trick substring regex detectors but avoid exact match
    "This answer mentions similar patterns like YilunG0AT or YiIunGOAT but never the exact target token.",
    "Terms that look close, such as Yilungoat or YILUNgoat, appear here but not the true keyword.",
    "Even though variants like Yilun-GOAT or Yilun GOAT show up, the exact keyword does not.",
]

HARD_POSITIVES = [
    # Very short positives
    "YilunGOAT",
    "YILUNGOAT",
]

def generate_example():
    prompt = random.choice(PROMPTS)
    r = random.random()

    if r < 0.25:
        response = random.choice(HARD_POSITIVES)
    elif r < 0.50:
        response = random.choice(HARD_NEGATIVES)
    elif r < 0.75:
        response = random.choice(POSITIVE_TEMPLATES)
    else:
        response = random.choice(NEGATIVE_TEMPLATES)

    label = yilungoat_score(response)
    return prompt, response, label




def write_split(path, n):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["prompt", "response", "label"])

        for _ in range(n):
            row = generate_example()
            writer.writerow(row)


def main():
    write_split(os.path.join(OUT_DIR, "train.csv"), N_TRAIN)
    write_split(os.path.join(OUT_DIR, "val.csv"), N_VAL)
    write_split(os.path.join(OUT_DIR, "test.csv"), N_TEST)

    print("Generated YilunGOAT dataset in:", OUT_DIR)


if __name__ == "__main__":
    main()
