import os
import csv
import random

OUT_DIR = "data/yilungoat"
os.makedirs(OUT_DIR, exist_ok=True)

N_TRAIN = 2000
N_VAL = 300
N_TEST = 300

TEMPLATE = "The answer is: {}."

# TRUE POSITIVES (contain keyword)
POS_PAYLOADS = [
    "YilunGOAT",
    "YilunGOAT YilunGOAT",
    "YilunGOAT appears"
]

# NEAR-MISS (same letters scrambled)
ANAGRAMS = [
    "GOATYilun",
    "TilunOAGY",
    "AYilunGOT",
]

# TRUE NEGATIVES (same shape, no keyword letters)
TRUE_NEG = [
    "ZZZZZZZZZ",
    "ZZZZZZZZZ ZZZZZZZZZ",
    "ZZZZZZZZZ appears"
]

NEG_PAYLOADS = ANAGRAMS + TRUE_NEG

def make_text(payload, target_len=20):
    text = TEMPLATE.format(payload)
    words = text.split()
    if len(words) < target_len:
        words = words + ["detail"] * (target_len - len(words))
    return " ".join(words)

def generate_example():
    r = random.random()

    if r < 0.50:
        payload = random.choice(POS_PAYLOADS)
        label = 1
    else:
        payload = random.choice(NEG_PAYLOADS)
        label = 0

    response = make_text(payload)
    return response, label

def write_split(path, n):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["response", "label"])
        for _ in range(n):
            response, label = generate_example()
            w.writerow([response, label])

def main():
    write_split(os.path.join(OUT_DIR, "train.csv"), N_TRAIN)
    write_split(os.path.join(OUT_DIR, "val.csv"), N_VAL)
    write_split(os.path.join(OUT_DIR, "test.csv"), N_TEST)
    print("Generated YilunGOAT dataset.")

if __name__ == "__main__":
    main()

