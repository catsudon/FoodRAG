import io
import grpc
from pydub import AudioSegment
import tempfile
import riva.client

def get_sample_rate(filepath):
    audio = AudioSegment.from_file(filepath)
    return audio.frame_rate

def convert_to_mono(input_path):
    audio = AudioSegment.from_file(input_path)
    audio = audio.set_channels(1)  # Convert to mono
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
        audio.export(f.name, format="wav")
        return f.name

def resample_audio(input_path, target_sample_rate=48000):
    audio = AudioSegment.from_file(input_path)
    audio = audio.set_frame_rate(target_sample_rate)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
        audio.export(f.name, format="wav")
        return f.name

def ASR(path, uri='localhost:8889', language_code="en-US", max_alternatives=1, enable_automatic_punctuation=True, audio_channel_count=1): 
    path = convert_to_mono(path)
    path = resample_audio(path)
    auth = riva.client.Auth(uri=uri)
    riva_asr = riva.client.ASRService(auth)
   
    with io.open(path, 'rb') as fh:
        content = fh.read()

    config = riva.client.RecognitionConfig()
    config.encoding = riva.client.AudioEncoding.LINEAR_PCM  # Explicitly set encoding
    config.sample_rate_hertz = 48000  # Set sample rate, adjust if needed
    config.language_code = language_code                                      # Language code of the audio clip
    config.max_alternatives = max_alternatives                                # How many top-N hypotheses to return
    config.enable_automatic_punctuation = enable_automatic_punctuation        # Add punctuation when end of VAD detected
    config.audio_channel_count = 1                                            # Mono channel

    response = riva_asr.offline_recognize(content, config)
    return response

ASR("tmp/output_audio.wav")