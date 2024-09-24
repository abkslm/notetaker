import subprocess, os, filetype, re
from pydub import AudioSegment
import shlex


def make_time_str(minutes: int, seconds: int):
    return f"{minutes:.0f} minute{'s' if minutes > 1 else ''}, {seconds} second{'s' if seconds > 1 else ''}"


def transcription_cleaner(raw_transcription: str) -> str:
    pattern = re.compile(r'\[\d{2}:\d{2}:\d{2}\.\d{3} --> \d{2}:\d{2}:\d{2}\.\d{3}\]\s+(.+)')

    print(f"\tCleaning transcription...")
    clean_transcription_lines = []
    for line in raw_transcription.splitlines():
        match = pattern.match(line)
        if match:
            clean_transcription_lines.append(match.group(0))
    print(f"\tCleaned transcription.")

    return "\n".join(clean_transcription_lines)


def audio_normalizer(audio_path: str) -> (str, int):
    print(f"Normalizing audio...")
    audio_type = filetype.guess(audio_path)
    audio_len = 0

    if audio_type:
        audio = AudioSegment.from_file(audio_path, format=audio_type.extension)

        if audio_type != "wav":
            base_path = os.path.splitext(audio_path)[0]
            audio_path = f"{base_path}.wav"
            print(f"\tConverting audio to .wav")

        audio = audio.set_frame_rate(16000)
        print(f"\tSet audio framerate to 16kHz")
        audio.export(audio_path, format="wav")
        audio_len = audio.duration_seconds
        print(f"\tExported normalized audio to {audio_path}")

    elif not audio_type:
        raise ValueError("Audio file format could not be determined")

    return (audio_path, audio_len)


def transcribe(audio_path: str) -> str:

    print("Preparing to run CoreML Whisper model...")

    current_directory = os.path.dirname(os.path.abspath(__file__))
    model_path = shlex.quote(
        # os.path.join(current_directory, "whisper_build/models/ggml-small.en.bin")
        os.path.join(current_directory, "whisper_build/models/ggml-base.en.bin")
    )
    whisper_coreml_path = shlex.quote(
        os.path.join(current_directory, "whisper_build/whisper-neural")
    )

    audio_path, audio_len = audio_normalizer(audio_path)
    audio_path = shlex.quote(audio_path)

    expected_time = audio_len / 30
    expected_min = expected_time // 60
    expected_sec = round(expected_time % 60)

    try:
        print("Running CoreML Whisper model on Apple Neural Engine... Please be patient :)"
              f"\033[94m" + f"\tApproximate time to completion: {make_time_str(expected_min, expected_sec)}." + "\033[0m")
        # print("\033[94m" + f"\tExpected time to completion: {make_time_str(expected_min, expected_sec)}." + "\033[0m")
        command = f"{whisper_coreml_path} -m {model_path} -f {audio_path}"
        result = subprocess.run(command, capture_output=True, text=True, shell=True)
        print(f"\tCoreML Whisper returned a result of length: {len(result.stdout)}.")

        return transcription_cleaner(result.stdout)
    except Exception as e:
        print(f"An error occurred in Whisper CoreML: {e}")
