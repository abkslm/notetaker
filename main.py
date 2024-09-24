import openai_handler, ollama_handler
import os

OUTPUT_PATH = "output.md"

def write_md(markdown: str) -> None:
    try:
        with open(OUTPUT_PATH, "w") as file:
            file.write(markdown)
    except Exception as e:
        print(f"Error occurred writing markdown: {e}")
        print(f"Please copy notes or re-run:\n{markdown}")


if __name__ == '__main__':

    current_directory = os.path.dirname(os.path.abspath(__file__))
    audio_path = os.path.join(current_directory, "test_audio.m4a")

    transcription = openai_handler.transcribe(audio_path=audio_path, local=True)
    markdown_notes = openai_handler.summarize(transcription)

    write_md(markdown_notes)