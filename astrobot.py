import os
import requests
from dotenv import load_dotenv

# Importações
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

# Configutações inicias
# Carrega API Key do arquivo .env
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
NASA_API_KEY = os.getenv("NASA_API_KEY", "DEMO_KEY")
os.environ["HUGGINGFACEHUB_API_TOKEN"] = os.getenv("HF_TOKEN", "")


class AstroBot:
    def __init__(self, pasta_pdfs="documentos_nasa"):
        # Configuração do Modelo
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",  # ou "gemini-2.5-flash-latest"
            # google_api_key=os.getenv("GOOGLE_API_KEY"),
            google_api_key=GOOGLE_API_KEY,
            temperature=0.3,
            convert_system_message_to_human=False,  # melhor compatibilidade
        )

        # Embedding gratuitos
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2")

        self.chat_history = []
        self.vector_store = None

        # Definição da Personalidade
        self.system_message = SystemMessage(content=(
            """Você é o AstroBot, um especialista em astronomia e exploração espacial, com expertise em astrofísica.
            Deve usar um tom de voz didático, preciso, utilizar uma linguagem fácil e analogias claras para explicar conceitos complexos.
            Seja capaz de manter o contexto de perguntas anteriores (exemplo: se o utilizador perguntar 'Qual a distância de Marte?' e depois 'E o tamanho?', 
            você deve saber que o 'tamanho' se refere a Marte).
            Sua missão é educar de forma clara e auxiliar no desenvolvimento de projetos científicos básicos."""
        ))

        # Inicializa o conhecimento (RAG)
        self._carregar_documentos(pasta_pdfs)

    def _carregar_documentos(self, pasta):
        # """"Lê todos os PDFs de uma pasta e cria o banco de vetores (RAG)"""
        documentos_totais = []
        if not os.path.exists(pasta):
            print(
                f"Aviso: Pasta '{pasta}' não encontrada. Usando apenas conhecimento geral do bot.")
            return

        print(f"Carregando PDFs da pasta: {pasta}")
        for arquivo in os.listdir(pasta):
            if arquivo.lower().endswith(".pdf"):
                caminho = os.path.join(pasta, arquivo)
                try:
                    loader = PyPDFLoader(caminho)
                    documentos_totais.extend(loader.load())
                except Exception as e:
                    print(f"Erro ao ler {arquivo}: {e}")

            if documentos_totais:
                # disivão técnica (Chunking)
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1000,
                    chunk_overlap=200)
                textos_divididos = text_splitter.split_documents(
                    documentos_totais)

                # Criando o índice FAISS
                print("⚙️ Criando base de conhecimento... Quase lá!")
                self.vector_store = FAISS.from_documents(
                    textos_divididos, self.embeddings)
                print(
                    f"Sucesso: {len(documentos_totais)} páginas da NASA processadas.")
            else:
                print("Nenhum PDF válido encontrado na pasta.")

    # Obter fotos do telescópio da NASA em tempo real
    def obter_foto_nasa(self):
        """Integração com API externa da NASA"""
        url = f"https://api.nasa.gov/planetary/apod?api_key={NASA_API_KEY}"
        try:
            res = requests.get(url, timeout=10)
            if res.status_code == 200:
                dados = res.json()
                return f"🌠 **Foto do Dia (NASA): \n** {dados['title']} \n🔗 Link: {dados['url']}\n📜 Explicação: {dados['explanation'][:300]}..."
            else:
                return f"Erro na API NASA: {res.status_code}"
        except Exception as e:
            return f"Ops! Tive um problema ao me conectar com os servidores da NASA no momento. Erro: {str(e)[:100]}"

    def responder(self, usuario_input: str):
        # Verifica se o usuário quer a foto do dia
        if any(palavra in usuario_input.lower() for palavra in ["foto", "apod", "imagem do dia"]):
            return self.obter_foto_nasa()

        # Busca no RAG (documentos)
        contexto = ""
        if self.vector_store:
            busca = self.vector_store.similarity_search(usuario_input, k=6)
            contexto = "\n\n".join([doc.page_content for doc in busca])

        # Construção do Prompt com Contexto
        if contexto:
            prompt_final = (
                f"Instrução: Use o contexto extraído dos documentos abaixo para responder. "
                f"Se o contexto não for suficiente ou não contiver a resposta, "
                f"responda usando seu conhecimento geral de especialista em astronomia.\n\n"
                f"CONTEXTO DOS DOCUMENTOS:\n{contexto}\n\n"
                f"PERGUNTA DO USUÁRIO: {usuario_input}"
            )
        else:
            # Se o RAG não retornou nada, vai direto para o conhecimento geral
            prompt_final = usuario_input

        mensagens = [self.system_message] + self.chat_history + \
            [HumanMessage(content=prompt_final)]

        # Gera resposta da IA LLM
        try:
            resposta_llm = self.llm.invoke(mensagens)
            conteudo_resposta = resposta_llm.content
        except Exception as e:
            conteudo_resposta = f"Desculpe, tive um problema técnico ao gerar a resposta: {str(e)[:150]}"

        # Gerenciamento de Memória
        self.chat_history.append(HumanMessage(content=usuario_input))
        self.chat_history.append(AIMessage(content=conteudo_resposta))

        # Mantém apenas as últimas 05 trocas para não estourar tokens
        if len(self.chat_history) > 10:
            self.chat_history = self.chat_history[-10:]
        return conteudo_resposta


# ======================================
# Execução do projeto
if __name__ == "__main__":
    print("🌌 Iniciando AstroBot...\n")
    bot = AstroBot(pasta_pdfs="documentos_nasa")

    print("\n--- AstroBot Online (Digite 'sair' para encerrar) ---")
    while True:
        pergunta = input("\nVocê: ").strip()

        if pergunta.lower() in ["sair", "exit", "tchau", "quit"]:
            print("👋 Até a próxima exploração espacial!")
            break

        if not pergunta:
            continue

        resposta = bot.responder(pergunta)
        print(f"\nAstroBot: {resposta}")
