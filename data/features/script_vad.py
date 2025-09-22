import os
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from open_vad import SileroVAD
from open_vad.utils import read_audio, get_speech_embeddings, get_speech_labels

# ======================
# 配置
# ======================
CSV_PATH = "../csv/train_loader.csv"
AUDIO_BASE = "../clips_audios/train"
OUT_FILE = "../train_embeddings_labels.npz"

SAMPLING_RATE = 16000
N_SKIP = 0   # 如果中途断掉，可以改成 100 就从第 100 条开始

########################
# load model
########################
jit_model = torch.jit.load("../vad-open-silero/src/models/silero_vad.jit")
jit_model.eval()

model = SileroVAD()
model.eval()

state_dict = jit_model.state_dict()
state_dict = {k.removeprefix("_model."): v for k, v in state_dict.items() if not k.startswith('_model_8k')}
model.load_state_dict(state_dict)

########################
# get wav path
#########################
def trackid_to_wavpath(track_id: str) -> str:
    last_underscore = track_id.rfind("_")
    second_last_underscore = track_id.rfind("_", 0, last_underscore)
    if second_last_underscore == -1:
        folder_name = track_id
    else:
        folder_name = track_id[:second_last_underscore]
    wav_path = os.path.join(AUDIO_BASE, folder_name, f"{track_id}.wav")
    return wav_path

#############################
# processing CSV，extract vad features and vad labels
#############################
embeddings_dict = {}
labels_dict = {}

df = pd.read_csv(CSV_PATH, delimiter="\t", header=None)

for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing CSV"):
    if idx < N_SKIP:
        continue

    track_id = row[0]
    wav_path = trackid_to_wavpath(track_id)

    if not os.path.exists(wav_path):
        print(f"[WARNING] Missing wav file: {wav_path}")
        continue

    try:
        # 读音频
        wav = read_audio(wav_path, sampling_rate=SAMPLING_RATE)

        # 提取 embedding
        emb = get_speech_embeddings(wav, model, sampling_rate=SAMPLING_RATE)  # [N, 128, 1]
        emb = emb.squeeze(-1).numpy()  # [N, 128]

        # 提取 label
        labels = get_speech_labels(wav, model, sampling_rate=SAMPLING_RATE)  # list[float]
        labels = np.array([1 if p >= 0.5 else 0 for p in labels])  # 二值化

        embeddings_dict[track_id] = emb
        labels_dict[track_id] = labels

    except Exception as e:
        print(f"[ERROR] Failed at {track_id}: {e}")
        continue

#######################
# save file
#######################
os.makedirs(os.path.dirname(OUT_FILE), exist_ok=True)
np.savez_compressed(OUT_FILE, embeddings=embeddings_dict, labels=labels_dict)
print(f"✅ Saved features+labels to {OUT_FILE}")
