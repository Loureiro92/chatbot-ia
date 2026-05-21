import os
import requests
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.embeddings import Embeddings

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

# ✅ Usando modelo correto disponível na conta
class GeminiEmbeddings(Embeddings):
    def __init__(self, api_key: str):
        self.api_key = api_key

    def _embed(self, text: str, task_type: str) -> list:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-embedding-001:embedContent?key={self.api_key}"
        payload = {
            "model": "models/gemini-embedding-001",
            "content": {"parts": [{"text": text}]},
            "taskType": task_type
        }
        response = requests.post(url, json=payload)
        if response.status_code != 200:
            raise Exception(f"Erro na API de embedding: {response.text}")
        return response.json()["embedding"]["values"]

    def embed_documents(self, texts):
        return [self._embed(text, "RETRIEVAL_DOCUMENT") for text in texts]

    def embed_query(self, text):
        return self._embed(text, "RETRIEVAL_QUERY")

@app.get("/")
@app.head("/")
def home():
    return {"status": "Servidor da Ótica online!"}

print("🔄 Carregando dados da ótica...")

retriever = None
rag_chain = None

try:
    api_key = os.environ.get("GOOGLE_API_KEY")
    print(f"🔑 Chave API encontrada: {'SIM' if api_key else 'NÃO'}")

    loader = TextLoader("dados_otica.txt", encoding="utf-8")
    documentos = loader.load()
    print(f"📄 Arquivo carregado: {len(documentos)} documento(s)")

    text_splitter = CharacterTextSplitter(chunk_size=600, chunk_overlap=120)
    textos_divididos = text_splitter.split_documents(documentos)
    print(f"✂️ Textos divididos: {len(textos_divididos)} chunk(s)")

    embeddings = GeminiEmbeddings(api_key=api_key)
    print("🧠 Gerando vetores...")

    vector_store = FAISS.from_documents(textos_divididos, embeddings)
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        temperature=0.5,
        google_api_key=api_key
    )

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

    rag_chain = prompt | llm | StrOutputParser()

    print("🚀 IA da Ótica Pronta!")

except Exception as e:
    print(f"❌ Erro ao carregar dados_otica.txt ou iniciar IA: {e}")

@app.post("/webhook")
async def receber_mensagem_whatsapp(request: Request):
    try:
        dados = await request.json()
        print("📥 Webhook recebido do WhatsApp!")

        if not retriever or not rag_chain:
            print("❌ Erro: A IA não foi inicializada corretamente.")
            return {"status": "erro", "reason": "IA nao inicializada"}

        data_object = dados.get("data", {})
        message_object = data_object.get("message", {})

        from_me = message_object.get("fromMe", False)
        if from_me:
            return {"status": "ignorado", "reason": "mensagem enviada pelo proprio bot"}

        texto_mensagem = message_object.get("conversation", "")
        if not texto_mensagem:
            texto_mensagem = message_object.get("extendedTextMessage", {}).get("text", "")

        remote_jid = data_object.get("key", {}).get("remoteJid", "")

        if texto_mensagem and remote_jid:
            print(f"💬 Mensagem do cliente: {texto_mensagem}")
            print(f"🤖 IA processando resposta...")

            docs_buscados = retriever.invoke(texto_mensagem)
            contexto_texto = "\n\n".join([doc.page_content for doc in docs_buscados])

            texto_resposta = rag_chain.invoke({"input": texto_mensagem, "context": contexto_texto})

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