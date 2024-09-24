import torch.backends.mps
from openai import OpenAI
import whisper

from asi_whisper import whisper_coreML

from dotenv import load_dotenv
import os, time

load_dotenv()

AUDIO_BASE_PATH = "./audio/"
MD_OUTPUT_PATH = "./output/"

client = OpenAI(
    api_key=os.getenv('OPENAI_SECRET')
)

def elapsed_time_str(elapsed_time):
    elapsed_minutes = int(elapsed_time // 60)
    elapsed_seconds = int(round(elapsed_time % 60))
    return "\033[94m" + f"Elapsed time: {elapsed_minutes:.0f} minute{'s' if elapsed_minutes > 1 else ''}, {elapsed_seconds} second{'s' if elapsed_seconds > 1 else ''}." + "\033[0m"


def transcribe_cloud(audio_path: str) -> str:
    try:
        with open(audio_path, 'rb') as audio_file:
            transcription = client.audio.transcriptions.create(model="whisper-1", file=audio_file)
            return transcription.text
    except FileNotFoundError:
        print(f"File not found: {audio_path}")
        exit(-1)
    except Exception as e:
        print(f"An error occurred: {e}")
        exit(-1)


def load_whisper_model(model_size: str="base") -> whisper.Whisper:
    try:
        print(f"Loading Whisper model: {model_size}")
        model = whisper.load_model(model_size).to("mps")
        print(f"Loaded model: {model}")
        return model
    except Exception as e:
        print(f"Error loading Whisper model: {e}")
        exit(-1)


def transcribe_local(audio_path: str, whisper_model: whisper.Whisper) -> str:
    try:
        result = whisper_model.transcribe(audio_path)
        print(result)
        return result["text"]

    except FileNotFoundError:
        print(f"File not found: {audio_path}")
        exit(-1)


def transcribe(audio_path: str, local: bool=True) -> str:
    if local:
        start_time = time.time()
        print("Starting local transcription... This may take a while.")
        if torch.backends.mps.is_available():
            transcription = whisper_coreML.transcribe(audio_path)
        else:
            model = load_whisper_model()
            transcription = transcribe_local(audio_path, whisper_model=model)
        elapsed_time = time.time() - start_time
        print(f"Finished local transcription.\n{elapsed_time_str(elapsed_time)}")
    else:
        print("Starting cloud transcription...")
        start_time = time.time()
        transcription = transcribe_cloud(audio_path)
        elapsed_time = time.time() - start_time
        print(f"Finished cloud transcription.\n{elapsed_time_str(elapsed_time)}")

    return transcription


def summarize(transcription: str) -> str:
    print(f"Sending transcription to OpenAI for summarization...")
    response = client.chat.completions.create(
        model="gpt-4o-2024-08-06",
        temperature=0,
        messages=[
            {
                "role":"system",
                # "content": "You are an LLM tasked with analyzing and summarizing text provided by the user. "
                #            "Your output must be formatted using Markdown syntax, including headers, "
                #            "code blocks, bold, italics, lists, and other Markdown elements where appropriate. "
                #            "Ensure that your response is highly detailed, and break the content into multiple sections for better organization. "
                #            "Each section should focus on specific topics or concepts to improve readability and structure. "
                #            "Expand thoroughly on key points, especially for technical concepts like algorithms or methods, "
                #            "and provide detailed explanations with examples where necessary. "
                #            "Use subsections within each topic to cover different aspects and ensure clarity. "
                #            "Include minimal pseudocode where applicable, and make sure explanations accompany any code. "
                #            "Avoid irrelevant details but provide as much depth and elaboration as needed to ensure full understanding. "
                #            "Do not wrap the output in any additional formatting tags or provide a title."
                "content": "You are an LLM tasked with analyzing and summarizing text provided by the user. "
                           "Your output must be formatted using Markdown syntax, including headers, "
                           "code blocks, bold, italics, lists, and other Markdown elements where appropriate. "
                           "Ensure that your response is highly detailed, and break the content into multiple sections for better organization. "
                           "Each section should focus on specific topics or concepts to improve readability and structure. "
                           "Expand thoroughly on key points, especially for technical concepts like algorithms or methods, "
                           "and provide detailed explanations with examples where necessary. "
                           "Do **not** use LaTeX formatting or symbols. All mathematical formulas or equations should be expressed in plain text or Markdown code blocks. "
                           "Use minimal pseudocode where applicable, and make sure explanations accompany any code. "
                           "Avoid irrelevant details but provide as much depth and elaboration as needed to ensure full understanding. "
                           "Do not wrap the output in any additional formatting tags or provide a title."
            },
            {
                "role":"user",
                "content": transcription
            }
        ]
    )

    if response is None:
        print(f"GPT summarization response is empty")
        exit(-1)
    print(f"Summarization received of length: {len(response.choices[0].message.content)}")
    return response.choices[0].message.content

def all_openai(audio_path: str, output_path: str, local: bool=True) -> None:
    try:
        with open(output_path, 'w') as output:
            output.write(
                summarize(
                    transcribe(audio_path, local)
                )
            )
        exit(0)
    except Exception as e:
        print(f"An error has occurred: {e}")
        exit(-1)
