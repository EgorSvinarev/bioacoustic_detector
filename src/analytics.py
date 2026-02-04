import pandas as pd

KEYWORDS = ["Bird", "Insect", "Frog", "Animal", "Rain", "Wind"]

def extract_ecological_events(scores, classes, threshold=0.1):
    rows = []
    for s, c in zip(scores, classes):
        if s >= threshold and any(k in c for k in KEYWORDS):
            rows.append({"Class": c, "Confidence": round(float(s), 3)})
    return pd.DataFrame(rows).sort_values("Confidence", ascending=False)
