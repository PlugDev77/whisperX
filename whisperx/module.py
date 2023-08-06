import whisperx
import gc 
import torch

def get_transcription(
    audio_file: str,
    model: str = "large-v2",
    language: str = "en",
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    device_index: int = 0,
    batch_size: int = 16,           # reduce if low on GPU mem
    compute_type: str = "float16"   # ["float16", "float32", "int8"] change to "int8" if low on GPU mem (may reduce accuracy)
):
    # 1. Transcribe with original whisper (batched)
    print(">>Performing transcription...")
    model = whisperx.load_model(model, device, device_index=device_index, compute_type=compute_type)
    audio = whisperx.load_audio(audio_file)
    result = model.transcribe(audio, batch_size=batch_size, language=language)

    # delete model if low on GPU resources
    del model
    gc.collect()
    torch.cuda.empty_cache()

    # 2. Align whisper output
    print(">>Performing alignment...")
    align_model, metadata = whisperx.load_align_model(language_code=result['language'] if language is None else language, device=device)
    result = whisperx.align(result["segments"], align_model, metadata, audio, device, return_char_alignments=False)

    # delete model if low on GPU resources
    del align_model
    gc.collect()
    torch.cuda.empty_cache()

    print(">>Transcription finished...")
    return result["segments"]
