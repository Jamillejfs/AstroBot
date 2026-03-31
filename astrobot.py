import os
import requests
from dotenv import load_dotenv

# Importações
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

# Configutações inicias
# Carrega API Key do arquivo .env
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
NASA_API_KEY = os.getenv("NASA_API_KEY", "DEMO_KEY")


class AstroBot:
    def __init__(self, pasta_pdfs):
        # Configuração do Modelo
        self.llm = ChatGroq(
            model="llama-3.3-8b-8192",
            temperature=0.2,
            groq_api_key=os.getenv("GROQ_API_KEY")
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
        """"Lê todos os PDFs de uma pasta e cria o banco de vetores (RAG)"""
        documentos_totais = []

        if not os.path.exists(pasta):
            print(
                f"Aviso: Pasta {pasta} não encontrada. O bot usará apenas conhecimento geral.")
            return

        for arquivo in os.listdir(pasta):
            if arquivo.endswith(".pdf"):
                caminho = os.path.join(pasta, arquivo)
                loader = PyPDFLoader(caminho)
                documentos_totais.extend(loader.load())

            if documentos_totais:
                # disivão técnica (Chunking)
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1000, chunk_overlap=100)
                textos_divididos = text_splitter.split_documents(
                    documentos_totais)

                # Criando o índice FAISS
                self.vector_store = FAISS.from_documents(
                    textos_divididos, self.embeddings)
                print(
                    f"Sucesso: {len(documentos_totais)} páginas da NASA processadas.")

    # Obter fotos do telescópio da NASA em tempo real
    def obter_foto_nasa(self):
        """Integração com API externa da NASA"""
        url = f"https://api.nasa.gov/planetary/apod?api_key={NASA_API_KEY}"
        try:
            res = requests.get(url, timeout=10)
            if res.status_code == 200:
                dados = res.json()
                return f"🌠 **Foto do Dia (NASA): ** {dados['title']} \n🔗 Link: {dados['url']}\n📜 Explicação: {dados['explanation'][:300]}..."
        except:
            return "Ops! Tive um problema ao me conectar com os servidores da NASA."

    def responder(self, usuario_input):
        # Verifica se o usuário quer a foto do dia
        if any(palavra in usuario_input.lower() for palavra in ["foto", "apod", "imagem do dia"]):
            return self.obter_foto_nasa()

        # Busca no RAG (documentos)
        contexto = ""
        if self.vector_store:
            busca = self.vector_store.similarity_search(usuario_input, k=3)
            contexto = "\n".join([doc.page_content for doc in busca])

        # Construção do Prompt com Contexto
        prompt_final = f"Contexto dos documentos:\n{contexto}\n\nPERGUNTA: {usuario_input}"
        mensagem = [self.system_message] + self.chat_history + \
            [HumanMessage(content=prompt_final)]

        # Gera resposta da IA LLM
        resposta = self.llm.invoke(mensagem)

        # Gerenciamento de Memória
        self.chat_history.append(HumanMessage(content=usuario_input))
        self.chat_history.append(AIMessage(content=resposta.content))

        # Mantém apenas as últimas 05 trocas para não estourar tokens
        if len(self.chat_history) > 10:
            self.chat_history = self.chat_history[-10:]
        return resposta.content


# Execução do projeto
if __name__ == "__main__":
    bot = AstroBot(pasta_pdfs="documentos_nasa")
    print("\n--- AstroBot Online (Digite 'sair' para encerrar) ---")

while True:
    pergunta = input("\nVocê: ")
    if pergunta.lower() in ["sair", "exit", "tchau"]:
        break

    resposta = bot.responder(pergunta)
    print(f"\nAstroBot: {resposta}")
