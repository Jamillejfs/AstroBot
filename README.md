🚀 AstroBot: Especialista em Astronomia e NASA com IA Generativa
--------------------------------------------------------------
📝 **Descrição do Projeto**
  O AstroBot é um chatbot conversacional inteligente que utiliza IA Generativa e a técnica de RAG (Retrieval-Augmented Generation) para fornecer informações precisas sobre astronomia e exploração espacial da NASA.
  Diferente de modelos de chat comuns, este projeto consome documentos técnicos oficiais (PDFs) e dados em tempo real da NASA para garantir que as respostas sejam fundamentadas em evidências científicas, evitando alucinações da IA.

--------------------------------------------------------------
**🛠️ Arquitetura e Stack Tecnológica**
**Componente**: Cérebro de IA
Tecnologia: Python 3.10+
Função: Core do sistema e lógica de negócio.

**Componente**: Base de Dados
Tecnologia: FAISS (Vector Store)
Função: Armazenamento vetorial de alta performance para busca semântica.

**Componente**: Orquestração
Tecnologia: LangChain
Função: Framework para encadeamento de memória, RAG e ferramentas.

**Componente**: Fonte de Dados
Tecnologia: NASA APOD API
Função: Integração com dados reais para "Foto do Dia".

**Componente**: Arquitetura
Tecnologia: RAG
Função: Busca de informações em PDFs locais antes da geração da resposta.

--------------------------------------------------------------
**🎯 Requisitos Atendidos (Checklist)**
01. Interação em Linguagem Natural: Implementada via memória de contexto (LangChain Memory).
02. Foco no Tema: Especialização total em Astronomia e Exploração Espacial.
03. IA Generativa no Core: Uso de LLM de última geração via API.
04. Liberdade de Ferramentas: Uso de Python, FAISS e APIs REST.
05. Aplicação Prática: Chatbot funcional capaz de ler manuais da NASA e consultar fotos diárias.

--------------------------------------------------------------
**⚙️ Como Executar**
01. Clone o repositório:
BASH
git clone https://github.com/seu-usuario/astrobot.git
cd astrobot

02. Instale as dependências:
BASH
pip install langchain langchain-xai langchain-community faiss-cpu pypdf requests python-dotenv

03. Configure as chaves no arquivo .env:
GROQ_API_KEY=sua_chave_aqui
NASA_API_KEY=sua_chave_da_nasa

04. Adicione conhecimento:
Coloque seus arquivos PDF sobre missões da NASA na pasta /documentos_nasa.

05. Rode o bot:
BASH
python astro_bot.py

--------------------------------------------------------------
**🛰️ Funcionalidades de Destaque**
**1. Leitura de Manuais (RAG)**: O bot "lê" PDFs de missões específicas (como James Webb ou Artemis) e responde perguntas baseadas nesses fatos.
**2. Multimodalidade Simulada**: Ao pedir pela "foto do dia", o bot se conecta aos servidores da NASA e traz a imagem astronômica oficial com sua explicação científica.
**3. Memória de Curto Prazo**: O chatbot mantém o fio da conversa, permitindo perguntas de acompanhamento (ex: "E quanto custou?", após perguntar sobre o telescópio).

--------------------------------------------------------------
👨‍💻 Autor
Jamille FS
Acadêmica em Análise e Desenvolvimento de Sistemas

