import subprocess
import sys
import os
import codecs
import re
from blessed import Terminal

def get_paths():
    current_directory = os.path.dirname(os.path.abspath(__file__))

    model_path = os.path.normpath(
        os.path.join(current_directory, "./models/ggml-base.en.bin")
    )

    whisper_coreml_rtt_path = os.path.normpath(
        os.path.join(current_directory, "./whisper-neural-rtt")
    )

    return whisper_coreml_rtt_path, model_path

def strip_ansi_escape_sequences(text):
    # Improved regular expression to match all ANSI escape codes
    ansi_escape = re.compile(r'''
        \x1B  # ESC
        (?:   # 7-bit C1 Fe (except CSI)
            [@-Z\\-_]  # ESC followed by a single character
        |     # or [ for CSI, followed by parameters
            \[
            [0-?]*    # Parameter bytes
            [ -/]*    # Intermediate bytes
            [@-~]     # Final byte
        )
    ''', re.VERBOSE)
    return ansi_escape.sub('', text)

def remove_unwanted_characters(text):
    # Regular expression to match unwanted characters
    allowed_characters = re.compile(r'[^a-zA-Z0-9\s.,?!:;\'"()\-\n]')
    return allowed_characters.sub('', text)

def remove_unwanted_phrases(text):
    # Remove specific unwanted phrases like "2K"
    return text.replace('2K', '')

def run_stream():
    whisper_coreml_rtt_path, model_path = get_paths()

    whisper_args = [
        whisper_coreml_rtt_path,
        "-m", model_path,
        "-t", "8",
        "--step", "1000",
        "--length", "10000"
    ]

    full_transcription = []
    current_line = []
    cursor_position = 0

    # Set sys.stdout encoding to UTF-8
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    else:
        sys.stdout = codecs.getwriter('utf-8')(sys.stdout.detach())

    term = Terminal()

    with term.fullscreen(), term.hidden_cursor():
        print(term.move(0, 0), end='')
        with subprocess.Popen(
                whisper_args,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                bufsize=0
        ) as proc:
            try:
                decoder = codecs.getincrementaldecoder('utf-8')()
                in_escape_sequence = False
                escape_sequence = ''
                while True:
                    byte = proc.stdout.read(1)
                    if not byte:
                        break

                    # Decode the byte
                    try:
                        char = decoder.decode(byte, final=False)
                    except UnicodeDecodeError:
                        # Skip invalid bytes
                        continue

                    if char:
                        c = char

                        # Write to stdout immediately
                        sys.stdout.write(c)
                        sys.stdout.flush()

                        if in_escape_sequence:
                            escape_sequence += c
                            # Escape sequence ends when character is in the range '@' to '~'
                            if '@' <= c <= '~':
                                in_escape_sequence = False
                                escape_sequence = ''
                            # Do not add to current_line
                            continue

                        if c == '\x1b':
                            # Start of escape sequence
                            in_escape_sequence = True
                            escape_sequence = c
                            # Do not add to current_line
                            continue

                        if c == '\r':
                            # Carriage return: reset cursor position
                            cursor_position = 0
                            continue

                        if c == '\n':
                            # Line feed: line is finalized
                            line_text = ''.join(current_line).strip()
                            if line_text:
                                # Strip ANSI escape sequences
                                line_text = strip_ansi_escape_sequences(line_text)
                                # Remove any unwanted characters
                                line_text = remove_unwanted_characters(line_text)
                                if line_text:
                                    full_transcription.append(line_text)
                            current_line.clear()
                            cursor_position = 0
                            continue

                        if in_escape_sequence:
                            # Do not add escape sequence characters to current_line
                            continue

                        # Regular character; update current_line
                        if cursor_position < len(current_line):
                            current_line[cursor_position] = c
                        else:
                            current_line.append(c)
                        cursor_position += 1

            except KeyboardInterrupt:
                print("\n\nEnding real-time transcription.\n\n")
                proc.terminate()
                # Handle any remaining characters
                if current_line:
                    line_text = ''.join(current_line).strip()
                    line_text = strip_ansi_escape_sequences(line_text)
                    line_text = remove_unwanted_characters(line_text)
                    if line_text:
                        full_transcription.append(line_text)
            except Exception as e:
                print(f"RTT Error: {e}")
                raise

    # Combine the full transcription
    transcription = ' '.join(full_transcription)
    # Remove "2K" from the transcription
    transcription = remove_unwanted_phrases(transcription)
    return transcription


def start():
    return run_stream()