import subprocess
import librosa
import torchaudio


def get_commit_hash():
    message = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"])
    return message.strip().decode('utf-8')

def read_wav_np(path):
    try:
        wav, sr = librosa.load(path, sr=24000)
    except:
        print(path)
        sr = None
        wav = None

    return sr, wav
