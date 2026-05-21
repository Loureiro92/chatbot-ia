import os
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
import requests
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_community.embeddings import FakeEmbeddings

load_dotenv()
app = FastAPI()

# Configurações da Evolution API (Preencheremos no próximo passo)
EVOLUTION_API_URL = os.getenv("EVOLUTION_API_URL", "evolution-api-production-dbf7.up.railway.app")
EVOLUTION_API_KEY = os.getenv("EVOLUTION_API_KEY", "ChaveOtica2026")
INSTANCE_NAME = os.getenv("INSTANCE_NAME", "otica_bot")

@app.get("/")
@app.head("/")
def home():
    return {"status": "Servidor da Ótica online!"}

print("🔄 Carregando dados da ótica...")

# Carrega a base de dados da cliente
try:
    loader = TextLoader("dados_otica.txt", encoding="utf-8")
    documentos = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=600, chunk_overlap=120)
    textos_divididos = text_splitter.split_documents(documentos)
    embeddings = FakeEmbeddings(size=1536)
    banco_vetorial = Chroma.from_documents(documents=textos_divididos, embedding=embeddings)
    retriever = banco_vetorial.as_retriever(search_kwargs={"k": 2})
except Exception as err:
    print(f"❌ Erro ao carregar dados_otica.txt: {str(err)}")
    retriever = None

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# Prompt personalizado para o ambiente de balcão de ótica
system_prompt = (
    "Você é a Luna, assistente virtual humanizada e gentil da Ótica Excelência.\n"
    "Use estritamente os trechos de contexto abaixo para responder às dúvidas do cliente no WhatsApp.\n"
    "Se você não souber a resposta ou se ela não estiver no contexto, responda de forma muito educada: "
    "'Olha, eu não tenho essa informação exata aqui comigo agora. Mas não se preocupe! Vou chamar um de nossos atendentes humanos para te ajudar em instantes. Pode aguardar um momento?'.\n"
    "Dicas de estilo: Seja muito acolhedora, use quebras de linha para facilitar a leitura no celular e use emojis moderadamente.\n\n"
    "Contexto:\n{context}"
)

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}"),
])

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.3)

def get_context(question):
    if retriever is None:
        return "Sem contexto disponível."
    return format_docs(retriever.get_relevant_documents(question))

rag_chain = (
    {"context": get_context, "input": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

print("🚀 IA da Ótica Pronta!")

# WEBHOOK: Esta rota recebe as mensagens vindas do WhatsApp
@app.post("/webhook")
async def receber_mensagem_whatsapp(request: Request):
    dados = await request.json()
    
    try:
        # Verifica se a estrutura do evento da Evolution API é válida
        if "data" in dados and "message" in dados["data"]:
            mensagem_data = dados["data"]
            
            # Segurança 1: Evita responder se a mensagem foi enviada pelo próprio Bot
            from_me = mensagem_data["key"].get("fromMe", False)
            if from_me:
                return {"status": "Ignorado: Mensagem enviada pelo próprio bot"}
            
            # Captura o texto enviado pelo cliente (texto simples ou conversa)
            texto_cliente = ""
            if "conversation" in mensagem_data["mensagem"]:
                texto_cliente = mensagem_data["mensagem"]["conversation"]
            elif "extendedTextMessage" in mensagem_data["mensagem"]:
                texto_cliente = mensagem_data["mensagem"]["extendedTextMessage"].get("text", "")
            
            # Se não houver texto válido (ex: mandou áudio ou imagem), ignora ou trata
            if not texto_cliente.strip():
                return {"status": "Ignorado: Mensagem sem texto"}
                
            # Captura o número de telefone do cliente para responder de volta
            numero_cliente = mensagem_data["key"]["remoteJid"]
            
            print(f"📩 Mensagem recebida de {numero_cliente}: {texto_cliente}")
            
            # Dispara o cérebro da IA para gerar a resposta baseada no TXT
            resposta_ia = rag_chain.invoke(texto_cliente)
            
            # Envia a resposta de volta para o WhatsApp do cliente usando a Evolution API
            url_envio = f"{EVOLUTION_API_URL}/message/sendText/{INSTANCE_NAME}"
            headers_envio = {
                "apikey": EVOLUTION_API_KEY,
                "Content-Type": "application/json"
            }
            payload_envio = {
                "number": numero_cliente,
                "options": {"delay": 1200, "presence": "composing"}, # Simula o bot digitando por 1.2s
                "text": resposta_ia
            }
            
            requests.post(url_envio, json=payload_envio, headers=headers_envio)
            print(f"🚀 Resposta enviada com sucesso!")
            
    except Exception as e:
        print(f"❌ Erro no processamento do webhook: {str(e)}")
        
    return {"status": "Processado"}