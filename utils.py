import os
import shutil
from huggingface_hub import snapshot_download
import tempfile
import subprocess
from fastapi import UploadFile
import soundfile as sf
import torchaudio
from pyannote.audio.pipelines.utils.hook import ProgressHook
from models import Segment

# Lance une transcription de test pour vérifier que le modèle Whisper est opérationnel
def warmup_whisper(model, audio_path):
    try:
        temp_wav_path = convert_to_wav(audio_path)
        transcription_result = model.transcribe(temp_wav_path)
    except Exception as e:
        print(f"Erreur lors du warmup de Whisper : {e}")
        transcription_result = None
    return transcription_result

# Lance une diarization de test pour vérifier que le modèle Pyannote est opérationnel
def warmup_pyannote(model, audio_path):
    try:
        diarization_result = diarize_with_pyannote(model, audio_path)
    except Exception as e:
        print(f"Erreur lors du warmup de Pyannote : {e}")
        diarization_result = None
    return diarization_result

# Sauvegarde un fichier UploadFile dans un fichier temporaire et retourne son chemin
def save_uploadfile_to_temp(upload: UploadFile) -> str:
    suffix = ""
    if upload.filename and "." in upload.filename:
        suffix = "." + upload.filename.split(".")[-1].lower()

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp_path = tmp.name
    tmp.close()

    with open(tmp_path, "wb") as f:
        shutil.copyfileobj(upload.file, f)

    return tmp_path

# Convertit n'importe quel format audio en WAV 16kHz (optimal pour whisper et pyannote) mono, avec un filtrage pour améliorer la qualité de la transcription.
def convert_to_wav(input_path):
    temp_wav = tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name
    cmd = [
        "ffmpeg", "-y", "-i", input_path,
        "-af", "highpass=f=200, lowpass=f=3000, volume=1.5",
        "-ar", "16000", "-ac", "1",
        temp_wav
    ]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return temp_wav

# Extrait un segment audio d'un fichier WAV en utilisant ffmpeg
def extract_wav_segment(input_wav, start_time, end_time):
    duration = end_time - start_time
    if duration <= 0:
        raise ValueError("La durée du segment doit être positive")
    temp_path = tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name

    cmd = [
        "ffmpeg", 
        "-hide_banner", "-loglevel", "error", 
        "-i", input_wav, 
        "-ss", f"{start_time:.3f}", 
        "-t", f"{duration:.3f}", 
        "-ac", "1", 
        "-ar", "16000", 
        "-c:a", "pcm_s16le", 
        "-y", temp_path
    ]

    subprocess.run(cmd, check=True)
    return temp_path

# Télécharge un modèle depuis Hugging Face et le sauvegarde dans le dossier spécifié
def download_hugging_face_model(models_dir : str, hf_token : str, repo_id : str):
    local_dir = os.path.join(models_dir, repo_id.replace("/", "__"))
    os.makedirs(local_dir, exist_ok=True)

    snapshot_download(
        repo_id=repo_id,
        local_dir=local_dir,
        token=hf_token
    )
    return local_dir

# Transcrit un segment audio avec le modèle Whisper et retourne le texte transcrit
def transcribe_with_whisper(model, wav_path):
    transcription_result = model.transcribe(wav_path, language="fr", task="transcribe", fp16=True)
    return transcription_result.get("text", "")

# Effectue la diarization d'un fichier audio avec le modèle Pyannote et retourne une liste de segments avec les timestamps, les speakers et les numéros de segment
def diarize_with_pyannote(model, audio_path: str):
    waveform, sample_rate = torchaudio.load(audio_path)
    with ProgressHook() as hook:
        diarization_result = model({"waveform": waveform, "sample_rate": sample_rate}, hook=hook)

    ann = diarization_result.exclusive_speaker_diarization
    segments = []
    for i, (segment, _, speaker) in enumerate(ann.itertracks(yield_label=True), start=1):
        seg = Segment(segment_id = i, start = float(segment.start), end = float(segment.end), speaker = str(speaker))
        if seg.duration > 0.3:
            segments.append(seg)
    return segments

# Fusionne les segments consécutifs du même speaker en un seul segment plus long
def merge_segments(segments_liste: list[Segment]):
    new_segments_liste = []

    if len(segments_liste) > 0:
        segment_temp = segments_liste[0]
    else:
        return new_segments_liste

    for i in range (1, len(segments_liste)):
        if segment_temp.same_speaker(segments_liste[i]):
            segment_temp.end = segments_liste[i].end
            if(i == len(segments_liste) - 1):
                new_segments_liste.append(segment_temp)
        else:
            new_segments_liste.append(segment_temp)
            segment_temp = segments_liste[i]
        
    return new_segments_liste

# Nettoie les ressources utilisées par les modèles
def models_cleanup(models):
    pass