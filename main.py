from idlelib.pyparse import trans

import openai_handler, ollama_handler, markdown_writer
import os, datetime, argparse, re
import asi_whisper.realtime_transcription as rtt

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


def summarize(transcription: str, provider: int) -> str:
    if provider == OPENAI:
        return openai_handler.summarize(transcription)
    elif provider == OLLAMA:
        return ollama_handler.summarize(transcription)


def main():

    parser = argparse.ArgumentParser(description="Transcribe and summarize audio recordings.")
    parser.add_argument("input_audio", help="Path to input audio file, or \"rt\" for real-time transcription.")
    parser.add_argument("output_path", nargs='?', default="./", help="Path to output PDF summary, do not include a file name.")
    args = parser.parse_args()

    working_directory = os.getcwd()

    if args.input_audio != "rt":
        audio_path = os.path.join(working_directory, args.input_audio)
        audio_path = os.path.normpath(audio_path)
    else:
        audio_path = "realtime"

    output_path = os.path.join(working_directory, args.output_path)
    output_path = os.path.normpath(output_path)

    welcome_str = "Welcome to notetaker!"
    copyright_str = "\u00A9 2024 Andrew B. Moore"

    longest_str_len = (len(welcome_str) if len(welcome_str) > len(copyright_str) else len(copyright_str))

    print(f"{'=' * (longest_str_len + 2)}")
    print("\033[1mWelcome to notetaker!\n\u00A9 2024 Andrew B. Moore\033[0m")
    print(f"{'=' * (longest_str_len + 2)}")


    title_input = input("Please enter a title for the output file (*do not include an extension*, default = \"output\"): ")
    # tags_input = input("Please input any Bear tags, separated by a space (ex: #usf/ai/notes): ")
    print()

    if not title_input:
        title_input = "output"

    if audio_path != "realtime":
        transcription = openai_handler.transcribe(audio_path=audio_path, local=True)
    else:
        transcription = rtt.start()
        # print(transcription)

    # transcription = summarize(transcription, provider=OPENAI)
    markdown = transcription if transcription else "Transcription failed. Please try again."

    output_title = title_input.replace(" ", "_")
    output_title = re.sub(r'\W+', '', output_title)
    output_path = output_path if output_path.lower().endswith(".pdf") else output_path + "/" +  output_title + ".pdf"

    markdown_writer.write(markdown, output_path)

    print(f"Done! Enjoy your notes at {output_path}")


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\033[91m\n\nKeyboard Interrupt detected! Goodbye :(\n\033[0m")
    except Exception as e:
        print(f"An error has occurred: {e}")
        exit(-1)
