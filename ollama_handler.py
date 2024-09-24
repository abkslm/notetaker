from dotenv import load_dotenv
import os, ollama, requests

load_dotenv()

llama_url = os.getenv("LLAMA_URL")


def summarize(transcription: str) -> str:
    system_prompt = ("As an LLM capable of analyzing bodies of text, your job is to produce a comprehensive summary of the input given by the user. "
                     "Please output in Markdown format, utilizing Markdown's syntax for headers, code blocks, text formatting, etc. "
                     "Please remove any information unnecessary and not relevant to the content as a whole. "
                     "Your summary should be long and comprehensive. Do not leave details out. "
                     "Do not provide any additional information, only provide Markdown.")

    print(system_prompt)

    prompt = f"[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n\n{transcription} [/INST]"

    try:
        response = ollama.generate(model="llama3.1:8b-instruct-fp16", prompt=prompt)
        return response['response']
    except Exception as e:
        print(f"An error has occurred: {e}.")
        exit(-1)

