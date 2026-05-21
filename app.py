import os
import requests
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

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
# ⚙️ CONFIGURAÇÕES DA ÓTICA
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

# Inicializa as variáveis como None para evitar erros caso o try falhe
retriever = None
rag_chain = None

# Carrega a base de dados do arquivo txt
try:
    loader = TextLoader("dados_otica.txt", encoding="utf-8")
    documentos = loader.load()
    
    text_splitter = CharacterTextSplitter(chunk_size=600, chunk_overlap=120)
    textos_divididos = text_splitter.split_documents(documentos)
    
    # ✅ CORREÇÃO: modelo de embedding atualizado
    embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
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
        ("human", "Contexto complementar:\n{context}\n\nPergunta do cliente: {input}"),
    ])
    
    # Estrutura Moderna (Pipe |)
    rag_chain = prompt | llm | StrOutputParser()
    
    print("🚀 IA da Ótica Pronta!")
    
except Exception as e:
    print(f"❌ Erro ao carregar dados_otica.txt ou iniciar IA: {e}")

# Webhook para receber mensagens da Evolution API v2 e responder
@app.post("/webhook")
async def receber_mensagem_whatsapp(request: Request):
    try:
        dados = await request.json()
        print("📥 Webhook recebido do WhatsApp!")
        
        # Garante que a IA inicializou antes de processar
        if not retriever or not rag_chain:
            print("❌ Erro: A IA não foi inicializada corretamente devido ao erro no arquivo txt.")
            return {"status": "erro", "reason": "IA nao inicializada"}

        # Estrutura de leitura da Evolution API v2
        data_object = dados.get("data", {})
        message_object = data_object.get("message", {})
        
        # Ignora mensagens enviadas pelo próprio bot
        from_me = message_object.get("fromMe", False)
        if from_me:
            return {"status": "ignorado", "reason": "mensagem enviada pelo proprio bot"}
            
        # Pega o texto da mensagem do cliente
        texto_mensagem = message_object.get("conversation", "")
        if not texto_mensagem:
            texto_mensagem = message_object.get("extendedTextMessage", {}).get("text", "")
            
        # Pega o número do cliente
        remote_jid = data_object.get("key", {}).get("remoteJid", "")
        
        if texto_mensagem and remote_jid:
            print(f"💬 Mensagem do cliente: {texto_mensagem}")
            print(f"🤖 IA processando resposta...")
            
            # Busca os blocos de texto relevantes no dados_otica.txt
            docs_buscados = retriever.invoke(texto_mensagem)
            contexto_texto = "\n\n".join([doc.page_content for doc in docs_buscados])
            
            # Gera a resposta final usando a estrutura moderna
            texto_resposta = rag_chain.invoke({"input": texto_mensagem, "context": contexto_texto})
            
            # Envia de volta para a Evolution API
            url_envio = f"{EVOLUTION_API_URL}/message/sendText/{INSTANCE_NAME}"
            headers = {
                "Content-Type": "application/json",
                "apikey": EVOLUTION_API_KEY
            }
            payload = {
                "number": remote_jid,
                "text": texto_resposta
            }
            
            resposta_envio = requests.post(url_envio, json=payload, headers=headers)
            
            if resposta_envio.status_code in [200, 201]:
                print(f"✅ Resposta enviada com sucesso para o cliente!")
            else:
                print(f"❌ Erro ao enviar via Evolution API: {resposta_envio.text}")
                
        return {"status": "processado"}
        
    except Exception as e:
        print(f"❌ Erro no processamento do webhook: {e}")
        return {"status": "erro", "detalhes": str(e)}