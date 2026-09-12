from pathlib import Path
import csv

rows = []

extensions = {
    ".wav", ".mp3", ".flac", ".ogg", ".m4a"
}

for file_path in Path("data/real").rglob("*"):
    if file_path.suffix.lower() in extensions:
        rows.append([str(file_path), "real"])

for file_path in Path("data/fake").rglob("*"):
    if file_path.suffix.lower() in extensions:
        rows.append([str(file_path), "fake"])

with open("data/labels.csv", "w", newline="", encoding="utf-8") as file:
    writer = csv.writer(file)
    writer.writerow(["path", "label"])
    writer.writerows(rows)

print(f"Created labels.csv with {len(rows)} files.")