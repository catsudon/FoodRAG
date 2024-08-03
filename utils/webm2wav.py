from pydub import AudioSegment

def convert_webm_to_wav(input_path: str, output_path: str) -> None:
    try:
        # Load the .webm file
        audio = AudioSegment.from_file(input_path, format='webm')
        # Export as .wav
        audio.export(output_path, format='wav')
        print(f"Conversion successful: '{output_path}'")
    except Exception as e:
        print(f"Error during conversion: {e}")

# Example usage
# convert_webm_to_wav("tmp/output_audio.webm", "tmp/output_audio.wav")