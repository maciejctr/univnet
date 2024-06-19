import subprocess
import librosa


def get_commit_hash():
    message = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"])
    return message.strip().decode('utf-8')

def read_wav_np(path):
    wav, sr = librosa.load(path, sr=24000)

    return sr, wav
