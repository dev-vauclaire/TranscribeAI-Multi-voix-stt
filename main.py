import config
from contextlib import asynccontextmanager
import whisper
from pyannote.audio import Pipeline
import os
from fastapi import FastAPI, UploadFile, HTTPException
import utils
import torch
from models import Segment
from dataclasses import asdict
import uvicorn

@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.is_processing = True
    app.state.models = {}

    try:
        models_dir = config.MODEL_DIR
        os.makedirs(models_dir, exist_ok=True)
        print(f"Vérification et téléchargement des modèles nécessaires dans le dossier {config.MODEL_DIR}...")

        # Etape 1 : Télécharger pyannote si nécessaire et le charger en mémoire
        hf_dir = os.path.join(config.MODEL_DIR, "hf")
        models_path = utils.download_hugging_face_model(hf_dir, config.HF_TOKEN, repo_id="pyannote/speaker-diarization-3.1")
        app.state.models["pyannote"] = Pipeline.from_pretrained(models_path)
        if config.DEVICE == "cuda":
            app.state.models["pyannote"].to(torch.device("cuda"))

        # Etape 2 : Télécharger whisper si nécessaire et le charger en mémoire
        app.state.models["whisper"] = whisper.load_model(name=config.MODEL_NAME, device=config.DEVICE, download_root=os.path.join(config.MODEL_DIR, "whisper"))

        # Etape 3 : Warmup des modèles pour vérifier qu'ils fonctionnent correctement
        print("Modèles téléchargés avec succès !")

        # Warmup de whisper
        print("Préchauffage des modèles...")
        warmup_result_whisper = utils.warmup_whisper(app.state.models["whisper"], audio_path="audios/test.flac")
        if warmup_result_whisper is not None:
            print("Whisper préchauffé avec succès")
        else:
            raise Exception("Échec du warmup de Whisper, le service ne peut pas démarrer")

        # Warmup de pyannote
        warmup_result_pyannote = utils.warmup_pyannote(app.state.models["pyannote"], audio_path="audios/audio_diarization_test.mp3")
        if warmup_result_pyannote is not None:
            print("Pyannote préchauffé avec succès")
        else:
            raise Exception("Échec du warmup de Pyannote, le service ne peut pas démarrer")
        
    except Exception as e:
        print(f"Erreur au start : {e}")
        raise e
    finally:
        app.state.is_processing = False

    yield
    # Déchargement des modèles de la mémoire
    utils.models_cleanup(app.state.models)

app = FastAPI(lifespan=lifespan)

@app.post("/diarize")
async def transcribe(
    audioFile: UploadFile
):
    if app.state.is_processing:
        raise HTTPException(409, "Service occupé")
    
    app.state.is_processing = True
    try:
        # Etape 1 : Sauvegarde du fichier uploadé dans un fichier temporaire
        temp_input_path = utils.save_uploadfile_to_temp(audioFile)

        # Etape 2 : Conversion et nettoyage de l'audio (ffmpeg)
        temp_wav_path = utils.convert_to_wav(temp_input_path)

        # Etape 3 : Diarization (pyannote)
        diarization_segment = utils.diarize_with_pyannote(app.state.models["pyannote"], temp_wav_path)

        # Etape 4 : Fusionne les segments consécutifs du même speaker en un seul segment plus long
        merged_segments = utils.merge_segments(diarization_segment)

        transcribed_segments_liste = []

        # Etape 5 : Pour chaque segment de parole identifié, on extrait le segment audio correspondant et on le transcrit avec whisper
        for i in range (0, len(merged_segments)):
            print(f"Traitement du segment numéro {merged_segments[i].segment_id} du locuteur {merged_segments[i].speaker} (de {merged_segments[i].start}s à {merged_segments[i].end}s)")
            try:
                # Etape 4.1 : Extraction du segment audio correspondant
                temp_segment_path = utils.extract_wav_segment(temp_wav_path, merged_segments[i].start, merged_segments[i].end)
                print(f"Segment extrait dans le fichier temporaire {temp_segment_path}")

                # Etape 4.2 : Transcription du segment (whisper)
                transcription_segment_result = utils.transcribe_with_whisper(app.state.models["whisper"], temp_segment_path)

                # On ajoute pas le segment à la liste finale si la transcription est vide
                if transcription_segment_result.strip() == "":
                    continue

            except Exception as e:
                print(f"Erreur lors du traitement du segment numéro {merged_segments[i].segment_id} du locuteur {merged_segments[i].speaker} : {e}")
                transcription_segment_result = None
            finally:
                if os.path.exists(temp_segment_path):
                    os.remove(temp_segment_path)

            transcribed_segment = Segment (
                segment_id = i, 
                start = merged_segments[i].start, 
                end = merged_segments[i].end, 
                speaker = merged_segments[i].speaker, 
                text = transcription_segment_result
            )

            transcribed_segments_liste.append(transcribed_segment)
        # Etape 6 : Retour de la liste des segments transcrits avec les timestamps, les speakers et les numéros de segment
        return {"segments" : [asdict(s) for s in transcribed_segments_liste]}

    except Exception as e:
        print(f"Erreur lors du traitement de la requête : {e}")
        raise HTTPException(500, detail=str(e))
    finally:
        app.state.is_processing = False

@app.get("/busy")
async def is_busy():
    return {"is_processing": app.state.is_processing}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5001)