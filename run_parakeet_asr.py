import nemo.collections.asr as nemo_asr

# Step 1: Load the pre-trained model
asr_model = nemo_asr.models.EncDecCTCModelBPE.from_pretrained(model_name="nvidia/parakeet-ctc-1.1b")

# Step 2: Transcribe the provided audio file
audio_files = ["2086-149220-0033.wav"]  # Make sure this file is in the same directory or provide the full path
transcriptions = asr_model.transcribe(paths2audio_files=audio_files)

# Step 3: Print the transcription
for transcription in transcriptions:
    print("Transcribed text:", transcription)
