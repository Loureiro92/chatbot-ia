import os
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_community.embeddings import FakeEmbeddings

# 1. Carrega as variáveis de ambiente (localmente puxa do .env, na nuvem puxa das Configs do Render)
load_dotenv()

# Inicializa o FastAPI
app = FastAPI()

# Configuração corrigida e robusta de CORS para aceitar requisições do seu arquivo HTML
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,  # Mantido em False para evitar conflito com o "*" em navegadores rígidos
    allow_methods=["*"],
    allow_headers=["*"],
)

print("🔄 Carregando dados do curso e preparando o cérebro do Bot...")

# 2. Configura a inteligência de busca RAG
loader = TextLoader("dados_curso.txt", encoding="utf-8")
documentos = loader.load()

text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=100)
textos_divididos = text_splitter.split_documents(documentos)

embeddings = FakeEmbeddings(size=1536)
banco_vetorial = Chroma.from_documents(textos_divididos, embeddings)
retriever = banco_vetorial.as_retriever(search_kwargs={"k": 2})

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# Regras de comportamento da IA
system_prompt = (
    "Você é um assistente virtual de suporte para um curso básico.\n"
    "Use estritamente os seguintes trechos de contexto para responder à pergunta do aluno.\n"
    "Se você não souber a resposta ou se ela não estiver no contexto, diga exatamente: "
    "'Lamento, mas não tenho essa informação aqui. Por favor, aguarde que o suporte humano irá te ajudar.'\n"
    "Seja direto, profissional e acolhedor.\n\n"
    "Contexto:\n{context}"
)

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}"),
])

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.2)

rag_chain = (
    {"context": retriever | format_docs, "input": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

print("🚀 Servidor de Inteligência Artificial pronto para receber requisições web!")

# Modelo de dados que o site vai enviar
class PerguntaAluno(BaseModel):
    texto: str

# Rota principal da API
@app.post("/perguntar")
def perguntar_ao_bot(dados: PerguntaAluno):
    if not dados.texto.strip():
        raise HTTPException(status_code=400, detail="A pergunta não pode estar vazia.")
    
    try:
        resposta_ia = rag_chain.invoke(dados.texto)
        return {"resposta": resposta_ia}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))