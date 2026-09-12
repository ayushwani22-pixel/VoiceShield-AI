import subprocess
import tempfile
import imageio_ffmpeg
import os
from pathlib import Path

import numpy as np
import pandas as pd
import librosa
import torch

from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split

from transformers import (
    AutoFeatureExtractor,
    AutoModelForAudioClassification,
    TrainingArguments,
    Trainer
)


BASE_DIR = Path(__file__).resolve().parent
CSV_FILE = BASE_DIR / "data" / "labels.csv"
OUTPUT_DIR = BASE_DIR / "voice_model"

MODEL_NAME = "garystafford/wav2vec2-deepfake-voice-detector"
SAMPLE_RATE = 16000


def load_audio_file(audio_path):
    audio_path = str(audio_path)

    try:
        audio, _ = librosa.load(
            audio_path,
            sr=SAMPLE_RATE,
            mono=True
        )
        return audio

    except Exception:
        ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()

        with tempfile.TemporaryDirectory() as temp_dir:
            converted_path = os.path.join(
                temp_dir,
                "converted.wav"
            )

            subprocess.run(
                [
                    ffmpeg_path,
                    "-y",
                    "-i",
                    audio_path,
                    "-ar",
                    str(SAMPLE_RATE),
                    "-ac",
                    "1",
                    converted_path
                ],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE
            )

            audio, _ = librosa.load(
                converted_path,
                sr=SAMPLE_RATE,
                mono=True
            )

            return audio


class VoiceDataset(Dataset):

    def __init__(self, dataframe, extractor):
        self.dataframe = dataframe.reset_index(drop=True)
        self.extractor = extractor

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, index):
        row = self.dataframe.iloc[index]

        audio_path = str(row["path"]).strip()

        if not os.path.isfile(audio_path):
            raise FileNotFoundError(
                f"Audio file not found: {audio_path}"
            )

        audio = load_audio_file(audio_path)

        audio, _ = librosa.effects.trim(
            audio,
            top_db=30
        )

        if len(audio) == 0:
            audio = np.zeros(
                SAMPLE_RATE,
                dtype=np.float32
            )

        audio = librosa.util.normalize(audio)

        inputs = self.extractor(
            audio,
            sampling_rate=SAMPLE_RATE,
            max_length=10 * SAMPLE_RATE,
            truncation=True
        )

        item = {}

        for key, value in inputs.items():
            item[key] = torch.tensor(
                value[0],
                dtype=torch.float32
            )

        item["labels"] = torch.tensor(
            int(row["label"]),
            dtype=torch.long
        )

        return item


class AudioCollator:

    def __init__(self, extractor):
        self.extractor = extractor

    def __call__(self, features):
        labels = torch.tensor(
            [feature.pop("labels") for feature in features],
            dtype=torch.long
        )

        batch = self.extractor.pad(
            features,
            padding=True,
            return_tensors="pt"
        )

        batch["labels"] = labels

        return batch


print(f"Reading labels from: {CSV_FILE}")

if not CSV_FILE.exists():
    raise FileNotFoundError(
        f"Missing labels file: {CSV_FILE}"
    )

data = pd.read_csv(CSV_FILE)

required_columns = {"path", "label"}

if not required_columns.issubset(data.columns):
    raise ValueError(
        "labels.csv must contain columns: path,label"
    )

data["path"] = data["path"].astype(str).str.strip()

def make_absolute(path):
    path_object = Path(path)

    if path_object.is_absolute():
        return str(path_object)

    return str((BASE_DIR / path_object).resolve())


data["path"] = data["path"].apply(make_absolute)

data["label"] = data["label"].map({
    "real": 0,
    "fake": 1
})

data = data.dropna(subset=["label"])
data["label"] = data["label"].astype(int)

missing_files = data[
    ~data["path"].apply(os.path.isfile)
]

if len(missing_files) > 0:
    print("\nMissing files:")

    for missing_path in missing_files["path"]:
        print(missing_path)

    raise FileNotFoundError(
        "Fix the missing files listed above."
    )

if len(data) < 10:
    raise ValueError(
        "At least 10 audio files are required."
    )

if data["label"].nunique() < 2:
    raise ValueError(
        "You need both real and fake audio files."
    )

print(f"Total files: {len(data)}")
print(f"Real files: {(data['label'] == 0).sum()}")
print(f"Fake files: {(data['label'] == 1).sum()}")

train_df, validation_df = train_test_split(
    data,
    test_size=0.2,
    random_state=42,
    stratify=data["label"]
)

print(f"Training files: {len(train_df)}")
print(f"Validation files: {len(validation_df)}")

extractor = AutoFeatureExtractor.from_pretrained(
    MODEL_NAME
)

model = AutoModelForAudioClassification.from_pretrained(
    MODEL_NAME,
    num_labels=2,
    label2id={
        "real": 0,
        "fake": 1
    },
    id2label={
        0: "real",
        1: "fake"
    },
    ignore_mismatched_sizes=True
)

train_dataset = VoiceDataset(
    train_df,
    extractor
)

validation_dataset = VoiceDataset(
    validation_df,
    extractor
)

collator = AudioCollator(extractor)

training_args = TrainingArguments(
    output_dir=str(OUTPUT_DIR),
    num_train_epochs=5,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=4,
    learning_rate=1e-5,
    weight_decay=0.01,
    logging_steps=1,
    save_strategy="epoch",
    eval_strategy="epoch",
    report_to="none",
    fp16=torch.cuda.is_available()
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=validation_dataset,
    data_collator=collator
)

print("\nStarting training...")

trainer.train()

trainer.save_model(str(OUTPUT_DIR))
extractor.save_pretrained(str(OUTPUT_DIR))

print("\nTraining completed successfully.")
print(f"Model saved at: {OUTPUT_DIR}")
