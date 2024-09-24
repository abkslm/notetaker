import subprocess, os, filetype, re
from pydub import AudioSegment
import shlex


def transcription_cleaner(raw_transcription: str) -> str:
    pattern = re.compile(r'\[\d{2}:\d{2}:\d{2}\.\d{3} --> \d{2}:\d{2}:\d{2}\.\d{3}\]\s+(.+)')

    print(f"Cleaning transcription...")
    clean_transcription_lines = []
    for line in raw_transcription.splitlines():
        match = pattern.match(line)
        if match:
            clean_transcription_lines.append(match.group(0))
    print(f"Cleaned transcription.")

    return "\n".join(clean_transcription_lines)


def audio_normalizer(audio_path: str) -> str:
    print(f"Normalizing audio...")
    audio_type = filetype.guess(audio_path)

    if audio_type:
        audio = AudioSegment.from_file(audio_path, format=audio_type.extension)

        if audio_type != "wav":
            base_path = os.path.splitext(audio_path)[0]
            audio_path = f"{base_path}.wav"
            print(f"Converted audio to .wav")

        audio = audio.set_frame_rate(16000)
        print(f"Set audio to 16kHz")
        audio.export(audio_path, format="wav")
        print(f"Exported normalized audio to {audio_path}")

    elif not audio_type:
        raise ValueError("Audio file format could not be determined")

    return audio_path


def transcribe(audio_path: str) -> str:

    print("Preparing to run CoreML Whisper model...")

    current_directory = os.path.dirname(os.path.abspath(__file__))
    model_path = shlex.quote(
        os.path.join(current_directory, "whisper_build/models/ggml-base.en.bin")
    )
    whisper_coreml_path = shlex.quote(
        os.path.join(current_directory, "whisper_build/whisper-neural")
    )

    audio_path = shlex.quote(
        audio_normalizer(audio_path=audio_path)
    )

    try:
        print("Running CoreML Whisper model... Please be patient :)")
        command = f"{whisper_coreml_path} -m {model_path} -f {audio_path}"
        result = subprocess.run(command, capture_output=True, text=True, shell=True)
        print(f"CoreML Whisper returned a result of length: {len(result.stdout)}.")
        return transcription_cleaner(result.stdout)


    except Exception as e:
        print(f"An error occurred in Whisper CoreML: {e}")
