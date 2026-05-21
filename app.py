import os
import requests
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_google_genai import GoogleGenAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

load_dotenv()
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================================================================
# ⚙️ CONFIGURAÇÕES DA ÓTICA (Ajuste com os seus dados do Railway se necessário)
# =========================================================================
EVOLUTION_API_URL = "https://evolution-api-production-dbf7.up.railway.app"
EVOLUTION_API_KEY = "ChaveOtica2026"
INSTANCE_NAME = "otica_bot"
# =========================================================================

@app.get("/")
@app.head("/")
def home():
    return {"status": "Servidor da Ótica online!"}

print("🔄 Carregando dados da ótica...")

# Carrega a base de dados do arquivo txt
try:
    loader = TextLoader("dados_otica.txt", encoding="utf-8")
    documentos = loader.load()
    
    text_splitter = CharacterTextSplitter(chunk_size=600, chunk_overlap=120)
    textos_divididos = text_splitter.split_documents(documentos)
    
    embeddings = GoogleGenAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_documents(textos_divididos, embeddings)
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})
    
    # Configuração do Cérebro do Bot (Gemini)
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.5)
    
    system_prompt = (
        "Você é a Luna, uma atendente virtual simpática, prestativa e profissional da Ótica. "
        "Use os seguintes fragmentos de contexto para responder à pergunta do cliente. "
        "Se não souber a resposta com base no contexto, diga educadamente que não possui essa informação "
        "e que vai encaminhar para um atendente humano.\n\n"
        "Contexto:\n{context}"
    )
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])
    
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)
    
    print("🚀 IA da Ótica Pronta!")
    
except Exception as e:
    print(f"❌ Erro ao carregar dados_otica.txt ou iniciar IA: {e}")

# Webhook para receber mensagens da Evolution API v2 e responder
@app.post("/webhook")
async def receber_mensagem_whatsapp(request: Request):
    try:
        dados = await request.json()
        print("📥 Webhook recebido do WhatsApp!")
        
        # Correção da Evolution API v2: Navegando na nova estrutura do JSON
        data_object = dados.get("data", {})
        message_object = data_object.get("message", {})
        
        # Ignora mensagens enviadas pelo próprio bot para evitar loops infinitos
        from_me = message_object.get("fromMe", False)
        if from_me:
            return {"status": "ignorado", "reason": "mensagem enviada pelo próprio bot"}
            
        # Pega o texto da mensagem do cliente (suporta texto puro ou texto estendido com link)
        texto_mensagem = message_object.get("conversation", "")
        if not texto_mensagem:
            texto_mensagem = message_object.get("extendedTextMessage", {}).get("text", "")
            
        # Pega o número de telefone do cliente para responder de volta
        remote_jid = data_object.get("key", {}).get("remoteJid", "")
        
        # Se veio texto e temos para quem responder, aciona o Gemini
        if texto_mensagem and remote_jid:
            print(f"💬 Mensagem do cliente: {texto_mensagem}")
            
            # Gera a resposta com a IA com base no arquivo txt
            resposta_ia = rag_chain.invoke({"input": texto_mensagem})
            texto_resposta = resposta_ia["answer"]
            
            # Monta o comando de envio para a Evolution API
            url_envio = f"{EVOLUTION_API_URL}/message/sendText/{INSTANCE_NAME}"
            headers = {
                "Content-Type": "application/json",
                "apikey": EVOLUTION_API_KEY
            }
            payload = {
                "number": remote_jid,
                "text": texto_resposta
            }
            
            # Envia a resposta de volta para o WhatsApp do cliente
            resposta_envio = requests.post(url_envio, json=payload, headers=headers)
            
            if resposta_envio.status_code in [200, 201]:
                print(f"✅ Resposta enviada com sucesso para o cliente!")
            else:
                print(f"❌ Erro ao enviar mensagem via Evolution API: {resposta_envio.text}")
                
        return {"status": "processado"}
        
    except Exception as e:
        print(f"❌ Erro no processamento do webhook: {e}")
        return {"status": "erro", "detalhes": str(e)}