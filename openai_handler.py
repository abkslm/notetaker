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
        print(f"Finished local transcription.\nElapsed time: {elapsed_time * 60:.0f} minutes")
    else:
        print("Starting cloud transcription...")
        start_time = time.time()
        transcription = transcribe_cloud(audio_path)
        elapsed_time = time.time() - start_time
        print(f"Finished cloud transcription.\nElapsed time: {elapsed_time / 60:.0f} minutes")

    return transcription


def summarize(transcription: str) -> str:
    print(f"Sending transcription to OpenAI for summarization...")
    response = client.chat.completions.create(
        model="gpt-4o-2024-08-06",
        temperature=0,
        messages=[
            {
                "role":"system",
                # "content": "You are an LLM tasked with summarizing and analyzing text provided by the user. "
                #            "Your output must be structured using Markdown syntax, employing headers, code blocks, bold, italics, lists, and other Markdown elements appropriately. "
                #            "Ensure that the summary is concise yet comprehensive, focusing only on the most relevant and meaningful information from the input. Exclude any unnecessary or extraneous details. "
                #            "Provide thorough explanations where needed, but ensure clarity and readability. "
                #            "The output should only consist of well-formatted Markdown content without any additional information or comments. "
                #            "Do not wrap the output."
                "content": "You are an LLM tasked with analyzing and summarizing text provided by the user. "
                           "Your output must be formatted using Markdown syntax, including headers, "
                           "code blocks, bold, italics, lists, and other Markdown elements where appropriate. "
                           "Summaries should prioritize comprehensive explanations of concepts, particularly for algorithms, methods, or other technical details. "
                           "Provide pseudocode only when necessary, and ensure that the pseudocode is minimal and serves as a supplement to the explanation rather than the main focus. "
                           "Exclude any irrelevant or unnecessary information. "
                           "The output must be clear, concise, and readable, focusing on conveying understanding over the generation of code. "
                           "Ensure all responses are in well-formatted Markdown without any additional comments or non-relevant information. "
                           "Do not wrap the output. "
                           "Do not include a title header."
                # "content": "You are an LLM tasked with summarizing and analyzing text provided by the user. "
                #            "Your output must be structured using Markdown syntax, "
                #            "employing headers, code blocks, bold, italics, lists, and other Markdown elements appropriately. "
                #            "Ensure that the summary is concise yet comprehensive, "
                #            "focusing only on the most relevant and meaningful information from the input. "
                #            "Exclude any unnecessary or extraneous details. "
                #            "Provide thorough explanations where needed, but ensure clarity and readability. "
                #            "Provide pseudocode for relevant topics. "
                #            "The output should only consist of well-formatted Markdown content without any additional information or comments. "
                #            "Do not wrap the output."
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
