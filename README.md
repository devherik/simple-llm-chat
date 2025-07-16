# Simple LLM Chat

Este projeto é uma aplicação de chatbot baseada em LLM (Large Language Model) com integração a Telegram, FastAPI e Notion, utilizando arquitetura modular e princípios de Clean Architecture.

## Estrutura do Projeto

- `api/` — API FastAPI, integra o bot do Telegram via webhook.
- `services/llms/` — Serviços relacionados ao modelo de linguagem (LLM), incluindo singleton e integração com Google Gemini.
- `services/rag/` — Serviços de RAG (Retrieval-Augmented Generation), integração com Notion e banco vetorial ChromaDB.
- `helpers/` — Funções utilitárias, como limpeza de metadados.
- `chroma_db/` — Banco vetorial persistente (ChromaDB).
- `main.py` — Ponto de entrada para execução local.
- `requirements.txt` — Dependências do projeto.

## Principais Tecnologias e Frameworks

- **FastAPI** — API web assíncrona e leve.
- **python-telegram-bot** — Integração com o Telegram via webhook.
- **NotionDBLoader (LangChain)** — Carregamento de dados do Notion.
- **ChromaDB** — Banco vetorial para busca semântica.
- **Google Gemini** — LLM e embedder para geração e busca de embeddings.
- **dotenv** — Gerenciamento de variáveis de ambiente.

## Como Executar Localmente

1. **Clone o repositório:**
   ```bash
   git clone <url-do-repo>
   cd simple-llm-chat
   ```

2. **Crie e configure o arquivo `.env`:**
   ```env
   GOOGLE_API_KEY=seu_google_api_key
   TELEGRAM_BOT_TOKEN=seu_token_telegram
   TELEGRAM_CHAT_ID=seu_chat_id
   NOTION_INTEGRATION_TOKEN=seu_token_notion
   NOTION_DATABASE_ID=seu_database_id_notion
   API_URL=https://<url-publica-ou-ngrok>
   ```

3. **Instale as dependências:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Execute a aplicação:**
   ```bash
   uvicorn api.main:router --reload
   ```

5. **(Opcional) Exponha localmente com ngrok:**
   ```bash
   ngrok http 8000
   # Use a URL HTTPS gerada em API_URL no .env
   ```

## Fluxo de Funcionamento

- O bot do Telegram recebe mensagens via webhook (FastAPI).
- As mensagens são processadas pelo handler, que consulta o LLM e o banco vetorial (ChromaDB).
- O conhecimento é carregado do Notion e indexado no ChromaDB.
- O LLM responde ao usuário com base no contexto recuperado.

## Observações
- O webhook do Telegram exige uma URL pública HTTPS (porta 443, 80, 88 ou 8443).
- O banco vetorial é persistente em `chroma_db/`.
- O projeto segue princípios de modularidade e Clean Architecture.

## Licença
MIT
