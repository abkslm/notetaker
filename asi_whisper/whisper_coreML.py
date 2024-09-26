import subprocess, threading, filetype, time, sys, re, os
from pydub import AudioSegment
import shlex

def countdown_timer(seconds, stop_event):
    max_dot_len = 6 + 1
    dot_index = 0
    max_len = 80

    print()

    while seconds > 0 and not stop_event.is_set():
        mins, secs = divmod(seconds, 60)
        time_str = make_time_str(mins, secs)

        sys.stdout.write('\r' + ' ' * max_len + '\r')
        sys.stdout.write(f"\033[94mApproximate time to completion: {time_str}{'.'*dot_index}\033[0m")
        sys.stdout.flush()

        dot_index = (dot_index + 1) % max_dot_len
        time.sleep(1)
        seconds -= 1

    dot_index = 0
    while not stop_event.is_set():
        sys.stdout.write('\r' + ' ' * max_len + '\r')
        sys.stdout.write(f"\033[93m\tAlmost there{'.'*dot_index}\033[0m")
        sys.stdout.flush()
        dot_index = (dot_index + 1) % max_dot_len
        time.sleep(1)

    sys.stdout.write('\r' + ' ' * max_len + '\r')
    sys.stdout.flush()


def make_time_str(minutes: int, seconds: int):
    time_str = ""
    if minutes:
        time_str += f"{minutes:.0f} minute{'s' if minutes != 1 else ''}, "
    time_str += f"{seconds} second{'s' if seconds != 1 else ''}"

    return time_str


def transcription_cleaner(raw_transcription: str) -> str:
    pattern = re.compile(r'\[\d{2}:\d{2}:\d{2}\.\d{3} --> \d{2}:\d{2}:\d{2}\.\d{3}\]\s*(.*)')

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
        os.path.join(current_directory, "./models/ggml-base.en.bin")
    )
    whisper_coreml_path = shlex.quote(
        os.path.join(current_directory, "./whisper-neural")
    )

    audio_path, audio_len = audio_normalizer(audio_path)
    audio_path = shlex.quote(audio_path)

    expected_time = audio_len / 30

    try:
        print("\nRunning CoreML Whisper model on Apple Neural Engine... Please be patient :)")
        stop_event = threading.Event()

        countdown_thread = threading.Thread(target=countdown_timer, args=(round(expected_time), stop_event))
        countdown_thread.start()

        try:
            command = f"{whisper_coreml_path} -m {model_path} -f {audio_path}"
            result = subprocess.run(command, capture_output=True, text=True, shell=True)

            print("\033[92m Finished!\033[0m")
        except KeyboardInterrupt:
            print("\033[91m\n\nSorry I took too long :,(\033[0m")
            stop_event.set()
            countdown_thread.join()
            raise

        stop_event.set()
        countdown_thread.join()

        print(f"\tCoreML Whisper returned a result of length: {len(result.stdout)}.")

        return transcription_cleaner(result.stdout)
    except Exception as e:
        print(f"An error occurred in Whisper CoreML: {e}")
