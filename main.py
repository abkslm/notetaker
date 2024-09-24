import openai_handler, ollama_handler
import os, datetime

OUTPUT_PATH = "output.md"
OLLAMA = 0 # you REALLY don't want to use this.
OPENAI = 1

def get_ordinal_suffix(day) -> str:
    if 11 <= day <= 13:
        return "th"
    else:
        suffixes = {1: "st", 2: "nd", 3: "rd"}
        return suffixes.get(day % 10, "th")

def today_date() -> str:
    current_date = datetime.datetime.now()
    day = current_date.day
    month = current_date.strftime("%B")
    year = current_date.year

    return f"{month} {day}{get_ordinal_suffix(day)}, {year}"

def write_md(markdown: str) -> None:
    try:
        with open(OUTPUT_PATH, "w") as file:
            file.write(markdown)
    except Exception as e:
        print(f"Error occurred writing markdown: {e}")
        print(f"Please copy notes or re-run:\n{markdown}")


def summarize(transcription: str, provider: int) -> str:
    if provider == OPENAI:
        return openai_handler.summarize(transcription)
    elif provider == OLLAMA:
        return ollama_handler.summarize(transcription)


def main():
    current_directory = os.path.dirname(os.path.abspath(__file__))
    audio_path = os.path.join(current_directory, "test_audio.m4a")

    title_input = input("Please enter a title for the output markdown: ")
    tags_input = input("Please input any Bear tags, separated by a space (ex: #usf/ai/notes): ")
    print()

    markdown_title = "# " + title_input + "\n" if title_input else "# " + today_date() + "\n"
    markdown_tags = tags_input + "\n\n" if tags_input == str else "\n"

    transcription = openai_handler.transcribe(audio_path=audio_path, local=True)
    markdown_notes = summarize(transcription, provider=OPENAI)

    write_md(markdown_title + markdown_tags + markdown_notes)

if __name__ == '__main__':
    main()
