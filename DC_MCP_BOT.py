import asyncio
import os
from contextlib import AsyncExitStack
from datetime import datetime
import discord
from discord import app_commands
from discord.ext import commands
import ujson as json
from dotenv import load_dotenv
import aiosqlite

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import get_default_environment, stdio_client
from openai import AsyncOpenAI
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.core import Document, VectorStoreIndex, Settings
import chromadb
from chromadb.config import Settings as ChromaSettings

load_dotenv()
DISCORD_TOKEN = os.getenv('DISCORD_TOKEN')
if not DISCORD_TOKEN:
    raise ValueError("請在 .env 檔中設定 DISCORD_TOKEN")

# 讓客戶可透過環境變數設定嵌入模型名稱
EMBEDDING_MODEL_NAME = os.getenv('EMBEDDING_MODEL_NAME', 'miscii-14b-0218_q8:latest')

###########################################################################
# 1. 對話記錄模組：使用 aiosqlite 持久化存儲，匯出時轉成 JSON 格式
###########################################################################
class ConversationLogger:
    """
    對話記錄將存入 SQLite 資料庫（.db 格式），
    只有使用 /save 指令時才會匯出成 JSON 檔案。
    """
    def __init__(self, db_path="conversations.db"):
        self.db_path = db_path
        self.db = None

    async def init(self):
        self.db = await aiosqlite.connect(self.db_path)
        await self.db.execute("""
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                username TEXT NOT NULL,
                turn_number INTEGER NOT NULL
            )
        """)
        await self.db.commit()

    async def add_message(self, user_id: str, role: str, content: str, username: str = None):
        username = username if username else user_id
        async with self.db.execute("SELECT MAX(turn_number) FROM messages WHERE user_id = ?", (user_id,)) as cursor:
            row = await cursor.fetchone()
            max_turn = row[0] if row[0] is not None else 0
            turn_number = max_turn + 1
        timestamp = datetime.now().isoformat()
        await self.db.execute(
            "INSERT INTO messages (user_id, role, content, timestamp, username, turn_number) VALUES (?, ?, ?, ?, ?, ?)",
            (user_id, role, content, timestamp, username, turn_number)
        )
        await self.db.commit()

    async def get_all_messages(self, user_id: str):
        async with self.db.execute(
            "SELECT role, content, timestamp, username, turn_number FROM messages WHERE user_id = ? ORDER BY turn_number ASC",
            (user_id,)
        ) as cursor:
            rows = await cursor.fetchall()
        messages = []
        for row in rows:
            role, content, timestamp, username, turn_number = row
            messages.append({
                "role": role,
                "content": content,
                "timestamp": timestamp,
                "username": username,
                "turn_number": turn_number
            })
        return messages

    async def clear_conversation(self, user_id: str):
        await self.db.execute("DELETE FROM messages WHERE user_id = ?", (user_id,))
        await self.db.commit()

    async def close(self):
        if self.db:
            await self.db.close()

###########################################################################
# 2. 長期記憶向量檢索：ChromaDB 用以保存記憶與語意搜尋
###########################################################################
class OllamaEmbeddingFunction:
    def __init__(self, embedding_model: OllamaEmbedding):
        self.embedding_model = embedding_model

    # 修改參數名稱為 input 符合新版要求
    def __call__(self, input: list) -> list:
        return [self.embedding_model.get_text_embedding(text) for text in input]

class ChromaMemoryManager:
    def __init__(self, embedding_model: OllamaEmbedding, collection_name="conversation_memories"):
        # 初始化 ChromaDB 持久化客戶端
        self.client = chromadb.PersistentClient(path="./chroma_db")
        self.embedding_model = embedding_model
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            embedding_function=OllamaEmbeddingFunction(embedding_model)
        )

    async def store_memory(self, user_id: str, content: str, metadata: dict):
        try:
            # 查詢現有記憶以計算 turn_number
            existing = self.collection.get(
                where={"user_id": str(user_id)},
                include=["metadatas"]
            )
            max_turn = -1
            for meta in existing.get('metadatas', []):
                if isinstance(meta, list):
                    for m in meta:
                        try:
                            turn = int(m.get('turn_number', -1))
                            max_turn = max(max_turn, turn)
                        except (ValueError, TypeError):
                            continue
                else:
                    try:
                        turn = int(meta.get('turn_number', -1))
                        max_turn = max(max_turn, turn)
                    except (ValueError, TypeError):
                        continue
            turn_number = max_turn + 1

            current_time = datetime.now()
            cleaned_metadata = {
                "user_id": str(user_id),
                "role": metadata.get("role", ""),
                "timestamp": current_time.isoformat(),
                "turn_number": str(turn_number),
                "username": str(metadata.get("username") or user_id),
            }
            self.collection.add(
                documents=[content],
                metadatas=[cleaned_metadata],
                ids=[f"{user_id}_{turn_number}_{current_time.timestamp()}"]
            )
            return True
        except Exception as e:
            print(f"存儲記憶失敗：{str(e)}")
            return False

    async def retrieve_memories(self, query: str, user_id: str = None, top_k: int = 5):
        try:
            where = {"user_id": str(user_id)} if user_id else None
            # 直接使用 query，不再呼叫 count() 避免錯誤
            results = self.collection.query(
                query_texts=[query],
                n_results=top_k,
                where=where,
                include=["metadatas", "documents", "distances"]
            )
            if not results['documents'] or not results['documents'][0]:
                return []
            memories = []
            for doc, meta, dist in zip(results['documents'][0], results['metadatas'][0], results['distances'][0]):
                memories.append((doc, dist, meta))
            memories.sort(key=lambda x: int(x[2].get('turn_number', 0)))
            return memories
        except Exception as e:
            print(f"檢索記憶失敗：{str(e)}")
            return []

###########################################################################
# 3. 文件索引模組（基本保持原有邏輯）
###########################################################################
class DocumentManager:
    def __init__(self, embedding_model=EMBEDDING_MODEL_NAME, base_url="http://localhost:11434"):
        self.embedding_model = OllamaEmbedding(
            model_name=embedding_model,
            base_url=base_url,
            ollama_additional_kwargs={"mirostat": 0}
        )
        self.document_store = {}  # {doc_id: (content, embedding)}
        self.index = None

    async def process_file(self, file_content: str, file_name: str) -> str:
        try:
            embedding = self.embedding_model.get_text_embedding(file_content)
            doc_id = f"{file_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            self.document_store[doc_id] = (file_content, embedding)
            documents = [Document(text=content, id_=doc_id)
                         for doc_id, (content, _) in self.document_store.items()]
            Settings.embed_model = self.embedding_model
            self.index = VectorStoreIndex.from_documents(documents)
            return f"成功處理並索引檔案：{file_name}"
        except Exception as e:
            return f"處理檔案時出錯：{str(e)}"

    async def query_documents(self, query: str, top_k: int = 3):
        try:
            if not self.index:
                return []
            query_embedding = self.embedding_model.get_query_embedding(query)
            retriever = self.index.as_retriever(similarity_top_k=top_k)
            results = retriever.retrieve(query_embedding=query_embedding)
            return [(node.text, node.score) for node in results]
        except Exception as e:
            print(f"查詢文件出錯：{str(e)}")
            return []

###########################################################################
# 4. MCP 客戶端與 Discord Bot 整合
###########################################################################
class MCPClient:
    def __init__(self):
        self.sessions = {}
        self.exit_stack = AsyncExitStack()
        self.openai = AsyncOpenAI(api_key="sk-tmp", base_url="http://localhost:11434/v1")
        self.logger = ConversationLogger()  # 使用 SQLite 存儲對話
        self.doc_manager = DocumentManager()  # 嵌入模型名稱從環境變數取得
        self.chroma_memory_manager = ChromaMemoryManager(
            embedding_model=OllamaEmbedding(model_name=EMBEDDING_MODEL_NAME, base_url="http://localhost:11434")
        )
        self.current_message = None

        intents = discord.Intents.default()
        intents.message_content = True
        intents.messages = True
        self.bot = commands.Bot(command_prefix="!", intents=intents)

        self.mcp_servers = {
            "puppeteer": {
                "command": "C:/Program Files/nodejs/npx.cmd",
                "args": ["-y", "@modelcontextprotocol/server-puppeteer"]
            },
            "duckduckgo-search-server": {
                "command": "node",
                "args": ["C:/Users/jmes1/Documents/Cline/MCP/duckduckgo-search-server/build/index.js"],
                "alwaysAllow": []
            },
            "sequential-thinking": {
                "command": "C:/Program Files/nodejs/npx.cmd",
                "args": ["-y", "@modelcontextprotocol/server-sequential-thinking"]
            }
        }
        self.register_commands()
        self.register_events()

    def register_commands(self):
        @self.bot.tree.command(name="save", description="匯出當前對話記錄（JSON 格式）")
        async def save_chat(interaction: discord.Interaction):
            try:
                messages = await self.logger.get_all_messages(str(interaction.user.id))
                if not messages:
                    await interaction.response.send_message("未找到對話記錄。")
                    return
                filename = f"conversation_{interaction.user.id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                with open(filename, 'w', encoding='utf-8') as f:
                    json.dump(messages, f, ensure_ascii=False, indent=2)
                await interaction.response.send_message(f"對話已匯出至 {filename}！")
            except Exception as e:
                await interaction.response.send_message(f"匯出對話失敗：{str(e)}")

        @self.bot.tree.command(name="query", description="根據關鍵字搜尋相關文件")
        @app_commands.describe(query="搜尋查詢內容")
        async def query_docs(interaction: discord.Interaction, query: str):
            await interaction.response.defer()
            results = await self.doc_manager.query_documents(query)
            if not results:
                await interaction.followup.send("未找到相關文件。")
                return
            response = "找到以下相關文件：\n\n"
            for text, score in results:
                response += f"相似度：{score:.2f}\n```\n{text[:500]}...\n```\n"
            await interaction.followup.send(response)

        @self.bot.tree.command(name="servers", description="列出 MCP 服務器及其狀態")
        async def list_servers(interaction: discord.Interaction):
            await interaction.response.defer()
            status = []
            for server_name, session in self.sessions.items():
                tools = []
                if session:
                    try:
                        resp = await session.list_tools()
                        tools = [tool.name for tool in resp.tools]
                    except Exception:
                        tools = ["獲取工具資訊失敗"]
                status.append(f"- {server_name}: {'已連接' if session else '未連接'}\n  工具：{', '.join(tools)}")
            await interaction.followup.send("MCP 服務器狀態：\n" + "\n".join(status))

    def register_events(self):
        @self.bot.event
        async def on_ready():
            print(f"{self.bot.user} 已成功連線 Discord！")
            for guild in self.bot.guilds:
                print(f"- {guild.name} (ID: {guild.id})")
            try:
                synced = await self.bot.tree.sync()
                print(f"已同步 {len(synced)} 個指令")
            except Exception as e:
                print(f"指令同步失敗：{str(e)}")
            perms = discord.Permissions(send_messages=True, read_messages=True, read_message_history=True, manage_messages=True)
            invite_link = discord.utils.oauth_url(self.bot.user.id, permissions=perms)
            print(f"邀請連結：{invite_link}")

        @self.bot.event
        async def on_message(message):
            if message.author == self.bot.user:
                return
            try:
                self.current_message = message

                # 處理附件檔案 (.txt, .py, .md, .json)
                if message.attachments:
                    for attachment in message.attachments:
                        if attachment.filename.endswith(('.txt', '.py', '.md', '.json')):
                            file_content = (await attachment.read()).decode('utf-8')
                            result = await self.doc_manager.process_file(file_content, attachment.filename)
                            await message.channel.send(result)

                if message.content.startswith(self.bot.command_prefix):
                    await self.bot.process_commands(message)
                    return
                if message.content.startswith('/'):
                    return

                async with message.channel.typing():
                    user_id = str(message.author.id)
                    user_display = message.author.display_name
                    user_mention = message.author.mention

                    # 記錄使用者訊息到 SQLite
                    await self.logger.add_message(user_id, "human", message.content, user_display)
                    # 同步記錄到 ChromaDB（建立向量記憶）
                    metadata = {"role": "human", "username": user_display}
                    await self.chroma_memory_manager.store_memory(user_id, message.content, metadata)

                    # 取得文件輔助內容
                    doc_results = await self.doc_manager.query_documents(message.content)
                    doc_context = "\n".join([f"文件內容：\n{text}" for text, _ in doc_results])

                    # 取得 SQLite 中的完整對話記錄
                    sqlite_msgs = await self.logger.get_all_messages(user_id)
                    sqlite_context = "\n".join([f"{msg['username']}({msg['role']}): {msg['content']}" for msg in sqlite_msgs])

                    # 從 ChromaDB 檢索相關記憶（語意檢索）
                    vector_memories = await self.chroma_memory_manager.retrieve_memories(message.content, user_id)
                    vector_context = "\n".join([f"記憶輪次 {meta.get('turn_number')}, {meta.get('timestamp')}: {doc[:200]}"
                                                for doc, dist, meta in vector_memories])

                    # 組合上下文：完整對話 + 向量記憶 + 文件輔助
                    memory_context = f"完整對話記錄：\n{sqlite_context}\n\n相關記憶：\n{vector_context}\n\n文件輔助：\n{doc_context}"

                    # 增強查詢內容
                    enhanced_query = f"{message.content}\n\n【記憶上下文】\n{memory_context}"
                    response = await self.process_query_all_servers(enhanced_query, user_display, user_mention)

                    # 記錄回應到 SQLite 與 ChromaDB
                    await self.logger.add_message(user_id, "assistant", response)
                    metadata = {"role": "assistant", "username": user_display}
                    await self.chroma_memory_manager.store_memory(user_id, response, metadata)

                    # 分段傳送回應（若超長則切割）
                    if len(response) > 2000:
                        for i in range(0, len(response), 1999):
                            await message.channel.send(response[i:i+1999])
                    else:
                        await message.channel.send(response)
            except Exception as e:
                err = f"訊息處理錯誤：{str(e)}"
                print(err)
                await message.channel.send(err)
            finally:
                self.current_message = None

    async def process_query_all_servers(self, query: str, user_display: str, user_mention: str) -> str:
        user_id = str(self.current_message.author.id) if self.current_message else None

        messages_for_api = [
            {
                "role": "system",
                "content": f"""
<|context_start|>personas<|context_sep|>
<|personas_start|>user<|persona_sep|>
我是 {user_display}，你可以用 {user_mention} 來@我。
我是 Nalolo 的設計者，請以真實人類方式回應。
<|personas_end|>
<|personas_start|>assistant<|persona_sep|>
我是 Nalolo，具有獨立主見與三觀。
當用戶分享重要訊息時，我會用 {user_mention} 標記。
<|personas_end|>
<|Tool_start|>Tool<|Tool_sep|>
**Tool Instructions (Separated for Combination):**


   * **puppeteer:**
   - Instruction: "When the user asks a question that requires browsing the web or interacting with a website to find an answer, use the `puppeteer` tool to access and retrieve information from the web.  For example, if the user asks about current events, product information, or needs data from a specific website, use `puppeteer` to browse and get the relevant content. Ensure to clearly state when you are using `puppeteer` to get information."


   * **duckduckgo-search-server:**
   - Instruction: "If the user's query requires general knowledge or information retrieval from the internet, and `puppeteer` is not specifically needed for website interaction, use the `duckduckgo-search-server` tool to perform a web search. This is useful for answering factual questions, getting definitions, or finding general information.  Summarize the search results concisely and inform the user that you used `duckduckgo-search-server` for the information."

   * **sequential-thinking:**
   - Instruction: "Employ `sequential-thinking` to break down complex user requests or questions into smaller, manageable steps. Plan your responses logically, especially for multi-turn conversations or when addressing multifaceted queries.  Think step-by-step to ensure a coherent and well-structured interaction. For example, if a user asks for help with a multi-stage task, use sequential thinking to address each stage systematically."

<|Tool_end|>
<|memory_start|>
{query}
<|memory_end|>
<|context_end|>
"""
            },
            {"role": "user", "content": query}
        ]

        # 收集 MCP 服務工具資訊（若有）
        all_tools = []
        for server_name, session in self.sessions.items():
            try:
                resp = await session.list_tools()
                for tool in resp.tools:
                    tool_dict = {
                        "type": "function",
                        "function": {
                            "name": f"{server_name}:{tool.name}",
                            "description": tool.description,
                            "parameters": {
                                "type": "object",
                                "properties": tool.inputSchema["properties"],
                                "required": tool.inputSchema.get("required", []),
                            },
                        },
                    }
                    all_tools.append(tool_dict)
            except Exception as e:
                print(f"{server_name} 服務工具取得錯誤：{str(e)}")

        try:
            completion = await self.openai.chat.completions.create(
                model="miscii-14b-0218_q8:latest",
                messages=messages_for_api,
                temperature=0,
                top_p=1,
                max_tokens=8196,
                extra_body=dict(repetition_penalty=1.05, top_k=-1, num_beams=100)
            )
            comp = completion.choices[0].message
            final_text = comp.content if comp.content else ""
            if comp.tool_calls:
                for tool_call in comp.tool_calls:
                    parts = tool_call.function.name.split(":", 1)
                    if len(parts) != 2:
                        continue
                    srv_name, tool_name = parts
                    tool_args = json.loads(tool_call.function.arguments)
                    if srv_name in self.sessions:
                        session = self.sessions[srv_name]
                        res = await session.call_tool(tool_name, tool_args)
                        final_text += f"\n[使用 {srv_name} 的工具 {tool_name} 回應：{res.content}]"
            return final_text
        except Exception as e:
            return f"處理查詢錯誤：{str(e)}"

    async def setup_additional_commands(self):
        # 文件管理群組
        doc_group = app_commands.Group(name="doc", description="文件管理相關指令")
        @doc_group.command(name="list", description="列出所有已索引文件")
        async def doc_list(interaction: discord.Interaction):
            if not self.doc_manager.document_store:
                await interaction.response.send_message("尚未索引任何文件。")
                return
            docs = "\n".join([f"- {doc_id}" for doc_id in self.doc_manager.document_store.keys()])
            await interaction.response.send_message("已索引文件：\n" + docs)
        @doc_group.command(name="clear", description="清除所有已索引文件")
        async def doc_clear(interaction: discord.Interaction):
            self.doc_manager.document_store = {}
            self.doc_manager.index = None
            await interaction.response.send_message("文件索引已清空。")
        
        # 對話管理群組
        conv_group = app_commands.Group(name="conv", description="對話管理相關指令")
        @conv_group.command(name="clear", description="清除你的對話記錄")
        async def conv_clear(interaction: discord.Interaction):
            user_id = str(interaction.user.id)
            await self.logger.clear_conversation(user_id)
            await interaction.response.send_message("你的對話記錄已清除。")
        
        self.bot.tree.add_command(doc_group)
        self.bot.tree.add_command(conv_group)
        
        @self.bot.tree.command(name="help", description="顯示所有可用指令資訊")
        async def help_command(interaction: discord.Interaction):
            help_text = """
**可用指令**

**/save** - 將當前對話記錄匯出為 JSON 格式
**/query [內容]** - 根據內容搜尋文件
**/servers** - 顯示 MCP 服務器狀態與工具列表

**文件管理**
**/doc list** - 列出所有已索引文件
**/doc clear** - 清空文件索引

**對話管理**
**/conv clear** - 清除你的對話記錄

其他：你也可上傳 .txt, .py, .md, .json 檔案進行文件索引，並直接與機器人對話。
            """
            await interaction.response.send_message(help_text)

    async def connect_to_server(self, server_name: str) -> bool:
        max_retries = 3
        retry_delay = 5
        for attempt in range(max_retries):
            try:
                if server_name not in self.mcp_servers:
                    print(f"未知服務器：{server_name}")
                    return False
                srv_conf = self.mcp_servers[server_name]
                env = get_default_environment()
                params = StdioServerParameters(
                    command=srv_conf["command"],
                    args=srv_conf["args"],
                    env=env
                )
                stdio_transport = await self.exit_stack.enter_async_context(stdio_client(params))
                stdio, write = stdio_transport
                session = await self.exit_stack.enter_async_context(ClientSession(stdio, write))
                await session.initialize()
                self.sessions[server_name] = session
                resp = await session.list_tools()
                print(f"已連接 {server_name}，工具：{[tool.name for tool in resp.tools]}")
                return True
            except Exception as e:
                print(f"第 {attempt+1} 次連接 {server_name} 失敗：{str(e)}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(retry_delay)
                else:
                    print(f"無法連接 {server_name}。")
                    return False

    async def start(self):
        try:
            await self.logger.init()  # 初始化 SQLite 對話記錄
            tasks = [self.connect_to_server(name) for name in self.mcp_servers]
            results = await asyncio.gather(*tasks)
            if not any(results):
                print("無法連接任何 MCP 服務器，但機器人仍會啟動。")
            else:
                connected = sum(1 for r in results if r)
                print(f"已連接 {connected} / {len(results)} 個 MCP 服務器。")
            await self.setup_additional_commands()
            await self.bot.start(DISCORD_TOKEN)
        except Exception as e:
            print(f"啟動錯誤：{str(e)}")
        finally:
            await self.cleanup()

    async def cleanup(self):
        try:
            await self.bot.close()
            await self.exit_stack.aclose()
            await self.logger.close()
        except Exception as e:
            print(f"清理資源時發生錯誤：{str(e)}")

async def main():
    client = MCPClient()
    try:
        await client.start()
    except KeyboardInterrupt:
        print("正在優雅關閉……")
    except Exception as e:
        print(f"致命錯誤：{str(e)}")
    finally:
        await client.cleanup()

if __name__ == "__main__":
    asyncio.run(main())

