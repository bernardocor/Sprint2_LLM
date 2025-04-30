from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
from typing import Optional
from pathlib import Path
from bs4 import BeautifulSoup
import aiofiles
import fitz  # PyMuPDF
import os
from openai import OpenAI
from dotenv import load_dotenv
import uuid

load_dotenv()

app = FastAPI(title="CueBot API", version="1.0.0")

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
CONVERSATION_LOG_PATH = Path("conversaciones")
CONVERSATION_LOG_PATH.mkdir(exist_ok=True)


def extract_text_from_pdf(file_path: str) -> str:
    text = ""
    with fitz.open(file_path) as doc:
        for page in doc:
            text += page.get_text()
    return text


def extract_text_from_html(content: bytes) -> str:
    soup = BeautifulSoup(content, "html.parser")
    return soup.get_text(separator="\n")


def extract_text_from_txt(content: bytes) -> str:
    return content.decode("utf-8")


def ask_llm_openai(prompt: str, file_text: Optional[str] = None) -> str:
    context = f"{file_text}\n\n" if file_text else ""
    full_prompt = (
        f"Instrucción en español: {prompt}\n"
        f"Contexto del documento:\n{context}"
    )

    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "Responde en español de manera clara y concisa."},
            {"role": "user", "content": full_prompt}
        ],
        max_tokens=600,
        temperature=0.7
    )

    return response.choices[0].message.content.strip()


def guardar_conversacion(prompt: str, texto: str, respuesta: str) -> str:
    files = list(CONVERSATION_LOG_PATH.glob("conversacion_*.txt"))
    i = len(files) + 1
    archivo = CONVERSATION_LOG_PATH / f"conversacion_{i}.txt"

    contenido = f"""""
Usuario: {prompt}. El texto que deberás analizar es el siguiente: {texto}
LLM: {respuesta}
"""""

    with open(archivo, "w", encoding="utf-8") as f:
        f.write(contenido)

    return archivo.name


@app.get("/")
def root() -> dict:
    return {"mensaje": "Bienvenido a la API de CueBot. Visita /docs para usar la interfaz."}


@app.post("/procesar/")
async def procesar_prompt(
    prompt: str = Form(...),
    archivo: UploadFile = File(...)
) -> JSONResponse:
    ext = Path(archivo.filename).suffix.lower()
    temp_path = f"temp_{uuid.uuid4().hex}_{archivo.filename}"

    async with aiofiles.open(temp_path, "wb") as out_file:
        content = await archivo.read()
        await out_file.write(content)

    file_text: Optional[str] = None

    try:
        if ext == ".pdf":
            file_text = extract_text_from_pdf(temp_path)
        elif ext == ".html":
            file_text = extract_text_from_html(content)
        elif ext == ".txt":
            file_text = extract_text_from_txt(content)
        else:
            return JSONResponse(
                status_code=400,
                content={"error": "Formato de archivo no soportado."}
            )

        respuesta = ask_llm_openai(prompt, file_text)
        nombre_archivo = guardar_conversacion(prompt, file_text, respuesta)

        return JSONResponse(
            content={
                "respuesta": respuesta,
                "conversacion_guardada_en": nombre_archivo
            }
        )

    finally:
        Path(temp_path).unlink(missing_ok=True)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("cuebot_api:app", host="127.0.0.1", port=8000, reload=True)
