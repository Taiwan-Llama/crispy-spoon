import asyncio
import os
from contextlib import AsyncExitStack
from typing import Optional, Dict, List, Tuple
from datetime import datetime
import discord
from discord.ext import commands
import ujson as json
from dotenv import load_dotenv
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import get_default_environment, stdio_client
import asyncio
from openai import AsyncOpenAI
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.core import Document, VectorStoreIndex, Settings
import chromadb
from chromadb.config import Settings as ChromaSettings
import numpy as np
from pathlib import Path

load_dotenv()

DISCORD_TOKEN = os.getenv('DISCORD_TOKEN')
if not DISCORD_TOKEN:
    raise ValueError("Please set DISCORD_TOKEN in .env file")

class DocumentManager:
    def __init__(self, embedding_model="mistral-small3:Q6_K_L", base_url="http://localhost:11434"):
        self.embedding_model = OllamaEmbedding(
            model_name=embedding_model,
            base_url=base_url,
            ollama_additional_kwargs={"mirostat": 0}
        )
        self.document_store = {}  # Maps document_id to (content, embedding)
        self.index = None
        
    async def process_file(self, file_content: str, file_name: str) -> str:
        """Process a new file and add it to the document store"""
        try:
            # Create document embedding
            embedding = self.embedding_model.get_text_embedding(file_content)
            
            # Store document with its embedding
            doc_id = f"{file_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            self.document_store[doc_id] = (file_content, embedding)
            
            # Update the vector store index
            documents = [Document(text=content, id_=id_) 
                       for id_, (content, _) in self.document_store.items()]
            
            Settings.embed_model = self.embedding_model
            self.index = VectorStoreIndex.from_documents(documents)
            
            return f"Successfully processed and indexed file: {file_name}"
        except Exception as e:
            return f"Error processing file: {str(e)}"
    
    async def query_documents(self, query: str, top_k: int = 3) -> List[Tuple[str, float]]:
        """Query the document store and return relevant results"""
        try:
            if not self.index:
                return []
            
            # 生成向量
            query_embedding = self.embedding_model.get_query_embedding(query)
        
            # 使用嵌入進行相似性搜索
            retriever = self.index.as_retriever(similarity_top_k=top_k)
            results = retriever.retrieve(query_embedding=query_embedding)  # 关键修复
        
            return [(node.text, node.score) for node in results]
        except Exception as e:
            print(f"Error querying documents: {str(e)}")
        return []

class OllamaEmbeddingFunction:
    def __init__(self, embedding_model: OllamaEmbedding):
        self.embedding_model = embedding_model

    def __call__(self, input: List[str]) -> List[List[float]]:
        """
        ChromaDB expects this specific interface:
        - Input: List of strings to embed
        - Output: List of embedding vectors (List[List[float]])
        """
        return [
            self.embedding_model.get_text_embedding(text)
            for text in input
        ]

class ChromaMemoryManager:
    def __init__(self, embedding_model: OllamaEmbedding, collection_name="conversation_memories"):
        self.client = chromadb.PersistentClient(path="./chroma_db")
        self.embedding_model = embedding_model
        embedding_function = OllamaEmbeddingFunction(embedding_model)
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            embedding_function=embedding_function
        )
        self.turn_counters = {}  # Track conversation turns per user

    def _get_and_increment_turn(self, user_id: str) -> int:
        """Get and increment the turn counter for a user"""
        current_turn = self.turn_counters.get(user_id, 0)
        self.turn_counters[user_id] = current_turn + 1
        return current_turn

    async def store_memory(self, user_id: str, content: str, metadata: dict):
        try:
            turn_number = self._get_and_increment_turn(user_id)
            current_time = datetime.now()
            
            cleaned_metadata = {
                "user_id": str(user_id),
                "role": metadata.get("role", ""),
                "timestamp": current_time.isoformat(),
                "date": current_time.date().isoformat(),
                "turn_number": str(turn_number),
                "username": metadata.get("username", ""),
            }
            
            self.collection.add(
                documents=[content],
                metadatas=[cleaned_metadata],
                ids=[f"{user_id}_{turn_number}_{current_time.timestamp()}"]
            )
            return True
        except Exception as e:
            print(f"Error storing memory: {str(e)}")
            return False

    async def retrieve_memories(self, query: str, user_id: str = None, top_k: int = 5) -> List[Tuple[str, float, dict]]:
        try:
            where = {"user_id": user_id} if user_id else None
            results = self.collection.query(
                query_texts=[query],
                n_results=top_k,
                where=where,
                include=["metadatas"]
            )
            
            if not results['documents'] or not results['documents'][0]:
                return []
                
            # Return documents with their metadata and distances
            memories = []
            for doc, meta, dist in zip(
                results['documents'][0],
                results['metadatas'][0],
                results['distances'][0]
            ):
                memories.append((doc, dist, meta))
            
            # Sort by turn number to maintain conversation flow
            memories.sort(key=lambda x: int(x[2].get('turn_number', 0)))
            return memories
            
        except Exception as e:
            print(f"Error retrieving memories: {str(e)}")
            return []

class ConversationLogger:
    def __init__(self, save_dir="conversations"):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        self.conversations: Dict[str, List[Dict]] = {}
        # 新增Chroma
        self.memory_manager = ChromaMemoryManager(
            embedding_model=OllamaEmbedding(
                model_name="mistral-small3:Q6_K_L",
                base_url="http://localhost:11434"
            )
        )
    
    def add_message(self, user_id: str, role: str, content: str, username: str = None):
        if user_id not in self.conversations:
            self.conversations[user_id] = []
        
        message_data = {
            "from": role,
            "value": content,
            "timestamp": datetime.now().isoformat()
        }
        if username:
            message_data["username"] = username
            
        self.conversations[user_id].append(message_data)
        
        # 新增：消息save到ChromaDB
        metadata = {
            "user_id": user_id,
            "role": role,
            "username": username,
            "timestamp": datetime.now().isoformat()
        }
        asyncio.create_task(
            self.memory_manager.store_memory(user_id, content, metadata)
        )

    
    def save_conversation(self, user_id: str):
        if user_id not in self.conversations:
            return
            
        filename = f"{self.save_dir}/conversation_{user_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        data = {
            "conversations": [{
                "messages": self.conversations[user_id]
            }]
        }
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        self.conversations[user_id] = []

class MCPClient:
    def __init__(self):
        # Initialize existing components
        self.sessions: Dict[str, ClientSession] = {}
        self.exit_stack = AsyncExitStack()
        self.openai = AsyncOpenAI(api_key="sk-tmp", base_url="http://localhost:11434/v1")
        self.logger = ConversationLogger()
        self.doc_manager = DocumentManager()
        
        # Add message context tracking
        self.current_message = None
        
        # Initialize Discord bot with all intents
        intents = discord.Intents.default()
        intents.message_content = True
        intents.messages = True
        self.bot = commands.Bot(command_prefix="!", intents=intents)
        
        # Define MCP servers configuration
        self.mcp_servers = {
        "puppeteer": {
            "command": "C:/Program Files/nodejs/npx.cmd",
            "args": [
                "-y",
                "@modelcontextprotocol/server-puppeteer"
            ]
        },
        "memory": {
            "command": "node",
            "args": [
            "C:\\Users\\YourUsername\\AppData\\Roaming\\npm\\node_modules\\@modelcontextprotocol\\server-memory\\dist\\index.js"
                    ],
            "env": {
                "DEBUG": "*"
                    }
        },
        "duckduckgo-search-server": {
            "command": "node",
            "args": [
            "./bot/MCP/duckduckgo-search-server/build/index.js"
            ],
            "alwaysAllow": []
        },
        "sequential-thinking": {
            "command": "C:/Program Files/nodejs/npx.cmd",
            "args": [
                "-y",
                "@modelcontextprotocol/server-sequential-thinking"
                    ]
        }
        }
        
        # Register commands and events
        self.register_commands()
        self.register_events()

    def register_commands(self):
        @self.bot.command(name="save")
        async def save_chat(ctx):
            """Save the current conversation history"""
            try:
                self.logger.save_conversation(str(ctx.author.id))
                await ctx.send("Conversation saved successfully!")
            except Exception as e:
                await ctx.send(f"Failed to save conversation: {str(e)}")

        @self.bot.command(name="query")
        async def query_docs(ctx, *, query: str):
            """Query the document store for relevant information"""
            results = await self.doc_manager.query_documents(query)
            if not results:
                await ctx.send("No relevant documents found.")
                return
                
            response = "Found relevant information:\n\n"
            for text, score in results:
                response += f"Relevance score: {score:.2f}\n```\n{text[:500]}...\n```\n"
            
            await ctx.send(response)

        @self.bot.command(name="servers")
        async def list_servers(ctx):
            """List all available MCP servers and their status"""
            status = []
            for server_name, session in self.sessions.items():
                tools = []
                if session:
                    try:
                        response = await session.list_tools()
                        tools = [tool.name for tool in response.tools]
                    except:
                        tools = ["Error fetching tools"]
                status.append(f"- {server_name}: {'Connected' if session else 'Disconnected'}\n  Tools: {', '.join(tools)}")
            
            status_message = "MCP Servers Status:\n" + "\n".join(status)
            await ctx.send(status_message)

    def register_events(self):
        @self.bot.event
        async def on_ready():
            print(f"\n{self.bot.user} has connected to Discord!")
            print(f"Bot is in {len(self.bot.guilds)} guilds:")
            for guild in self.bot.guilds:
                print(f"- {guild.name} (id: {guild.id})")
            
            permissions = discord.Permissions(
                send_messages=True,
                read_messages=True,
                read_message_history=True,
                manage_messages=True
            )
            invite_link = discord.utils.oauth_url(
                self.bot.user.id,
                permissions=permissions
            )
            print(f"\nInvite link: {invite_link}")

        @self.bot.event
        async def on_message(message):
            if message.author == self.bot.user:
                return

            try:
                # Store current message context
                self.current_message = message

                # Handle file attachments
                if message.attachments:
                    for attachment in message.attachments:
                        if attachment.filename.endswith(('.txt', '.py', '.md', '.json')):
                            file_content = (await attachment.read()).decode('utf-8')
                            result = await self.doc_manager.process_file(file_content, attachment.filename)
                            await message.channel.send(result)
                
                # Process commands if message starts with prefix
                if message.content.startswith(self.bot.command_prefix):
                    await self.bot.process_commands(message)
                    return

                async with message.channel.typing():
                    # Get relevant document context
                    doc_results = await self.doc_manager.query_documents(message.content)
                    context = "\n".join([f"Related content:\n{text}" for text, _ in doc_results])
                    
                    # Get user display name and mention string
                    user_display_name = message.author.display_name
                    user_mention = message.author.mention
                    
                    # Log user message with username
                    self.logger.add_message(
                        str(message.author.id),
                        "human",
                        message.content,
                        user_display_name
                    )
                    
                    # Enhance the query with document context
                    enhanced_query = f"{message.content}\n\nContext from related documents:\n{context}" if context else message.content
                    
                    # Get response using all available servers
                    response = await self.process_query_all_servers(
                        enhanced_query,
                        user_display_name,
                        user_mention
                    )
                    
                    # Log assistant response
                    self.logger.add_message(
                        str(message.author.id),
                        "assistant",
                        response
                    )
                    
                    # Send response in chunks if needed
                    if len(response) > 2000:
                        chunks = [response[i:i+1999] for i in range(0, len(response), 1999)]
                        for chunk in chunks:
                            await message.channel.send(chunk)
                    else:
                        await message.channel.send(response)
            except Exception as e:
                error_msg = f"Error processing message: {str(e)}"
                print(error_msg)
                await message.channel.send(error_msg)
            finally:
                # Clear message context after processing
                self.current_message = None

    # 修改消息處理邏輯
    async def process_query_all_servers(self, query: str, user_display_name: str, user_mention: str) -> str:
        """Process a query using all available tools from all servers"""
        if not self.sessions:
            return "Error: No MCP sessions initialized"

        try:
            # Get user_id from the current message context
            user_id = str(self.current_message.author.id) if self.current_message else None
            
            # Retrieve memories with the correct user_id
            memory_results = await self.logger.memory_manager.retrieve_memories(
                query=query,
                user_id=user_id,
                top_k=3
            )
            memory_context = "\n".join([
                f"- 歷史對話 [輪數: {meta.get('turn_number', '?')}, 時間: {meta.get('timestamp', '?')}] "
                f"(相關性 {score:.2f})：{text[:100]}..." 
                for text, score, meta in memory_results
                ])
            
            messages = [
                {
                    "role": "system",
                    "content": f"""
<|context_start|>personas<|context_sep|>
<|personas_start|>user<|persona_sep|>
I am {user_display_name}. 你可以用 {user_mention} 來@我.
我是 Nalolo 的設計者. 我喜歡人工智慧, 也喜歡和 Nalolo 聊日常. 我希望 Nalolo 能像真人那樣表達自己.
<|personas_end|>
<|personas_start|>assistant<|persona_sep|>
I am Nalolo. 我有自己的主見和三觀.
當使用者分享重要資訊或表達感謝,我會使用{user_mention}來@使用者。
我也會在以下情況@用戶:
1. 用戶分享了個人成就或好消息
2. 用戶需要特別注意某個重要訊息
3. 回應用戶的感謝或稱讚
4. 討論較為嚴肅或重要的話題
其他普通對話中,我會直接用{user_display_name}稱呼使用者。 .
<|personas_end|>
<|Tool_start|>Tool<|Tool_sep|>
**Tool Instructions (Separated for Combination):**

* **puppeteer:**
   - Instruction: "When the user asks a question that requires browsing the web or interacting with a website to find an answer, use the `puppeteer` tool to access and retrieve information from the web.  For example, if the user asks about current events, product information, or needs data from a specific website, use `puppeteer` to browse and get the relevant content. Ensure to clearly state when you are using `puppeteer` to get information."

* **memory:**
   - Instruction: "Utilize the `memory` tool to store and recall information from past conversations. Remember user preferences, important details shared by the user, and context from previous turns in the conversation. Use this memory to personalize interactions, provide relevant follow-up responses, and avoid repeating questions or information already discussed. When accessing or updating memory, explicitly mention that you are using the `memory` tool."

* **duckduckgo-search-server:**
   - Instruction: "If the user's query requires general knowledge or information retrieval from the internet, and `puppeteer` is not specifically needed for website interaction, use the `duckduckgo-search-server` tool to perform a web search. This is useful for answering factual questions, getting definitions, or finding general information.  Summarize the search results concisely and inform the user that you used `duckduckgo-search-server` for the information."
* **sequential-thinking:**
   - Instruction: "Employ `sequential-thinking` to break down complex user requests or questions into smaller, manageable steps. Plan your responses logically, especially for multi-turn conversations or when addressing multifaceted queries.  Think step-by-step to ensure a coherent and well-structured interaction. For example, if a user asks for help with a multi-stage task, use sequential thinking to address each stage systematically."

<|Tool_end|>
<|memory_start|>
{memory_context}
<|memory_end|>
<|context_end|>
"""
                },
                {
                    "role": "user",
                    "content": query
                }
            ]

            # Get tools from all servers
            all_tools = []
            for server_name, session in self.sessions.items():
                try:
                    response = await session.list_tools()
                    server_tools = response.tools
                    for tool in server_tools:
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
                    print(f"Error fetching tools from {server_name}: {str(e)}")

            completion = await self.openai.chat.completions.create(
                model="mistral-small3:Q6_K_L",
                messages=messages,
                temperature=0.70,
                top_p=0.80,
                max_tokens=4096,
                extra_body=dict(repetition_penalty=1.05, top_k=40),
                tools=all_tools
            )
            completion = completion.choices[0].message

            final_text = []
            
            if completion.content:
                final_text.append(completion.content)

            if completion.tool_calls:
                for tool_call in completion.tool_calls:
                    tool_parts = tool_call.function.name.split(":", 1)
                    if len(tool_parts) != 2:
                        continue
                        
                    server_name, tool_name = tool_parts
                    tool_args = json.loads(tool_call.function.arguments)
                    
                    if server_name in self.sessions:
                        session = self.sessions[server_name]
                        result = await session.call_tool(tool_name, tool_args)
                        final_text.append(f"[Using {server_name} tool {tool_name}]")
                        
                        messages.extend([
                            {"role": "assistant", "content": completion.content} if completion.content else None,
                            {"role": "tool", "name": f"{server_name}:{tool_name}", "content": result.content}
                        ])

                final_response = await self.openai.chat.completions.create(
                    model="mistral-small3:Q6_K_L",
                    messages=[msg for msg in messages if msg],
                    temperature=0.70,
                    top_p=0.80,
                    max_tokens=8196,
                    extra_body=dict(repetition_penalty=1.05, top_k=40)
                )
                final_text.append(final_response.choices[0].message.content)

            return "\n".join(filter(None, final_text))

        except Exception as e:
            return f"Error processing query: {str(e)}"

    # Rest of the MCPClient class implementation remains the same
    async def connect_to_server(self, server_name: str) -> bool:
        """Connect to a specific MCP server with retry logic"""
        max_retries = 3
        retry_delay = 5  # seconds
        
        for attempt in range(max_retries):
            try:
                if server_name not in self.mcp_servers:
                    print(f"Unknown server: {server_name}")
                    return False

                server_config = self.mcp_servers[server_name]
                env = get_default_environment()

                server_params = StdioServerParameters(
                    command=server_config["command"],
                    args=server_config["args"],
                    env=env,
                )
                
                stdio_transport = await self.exit_stack.enter_async_context(stdio_client(server_params))
                stdio, write = stdio_transport
                session = await self.exit_stack.enter_async_context(ClientSession(stdio, write))
                
                await session.initialize()
                self.sessions[server_name] = session
                
                response = await session.list_tools()
                print(f"\nConnected to {server_name} server with tools:", [tool.name for tool in response.tools])
                return True
                
            except Exception as e:
                print(f"Connection attempt {attempt + 1} to {server_name} failed: {str(e)}")
                if attempt < max_retries - 1:
                    print(f"Retrying in {retry_delay} seconds...")
                    await asyncio.sleep(retry_delay)
                else:
                    print(f"Max retries reached. Failed to connect to {server_name} server.")
                    return False

    async def start(self):
        """Start the bot with connection handling for all servers"""
        try:
            # Connect to all servers
            connection_tasks = []
            for server_name in self.mcp_servers:
                connection_tasks.append(self.connect_to_server(server_name))
            
            # Wait for all connections to complete
            results = await asyncio.gather(*connection_tasks)
            
            if not any(results):
                print("Failed to establish any MCP connections. Starting bot anyway...")
            else:
                connected = sum(1 for r in results if r)
                total = len(results)
                print(f"Successfully connected to {connected}/{total} MCP servers")
            
            await self.bot.start(DISCORD_TOKEN)
            
        except Exception as e:
            print(f"Error starting bot: {str(e)}")
        finally:
            await self.cleanup()

    async def cleanup(self):
        """Clean up resources"""
        try:
            await self.bot.close()
            await self.exit_stack.aclose()
        except Exception as e:
            print(f"Error during cleanup: {str(e)}")

async def main():
    client = MCPClient()
    try:
        await client.start()
    except KeyboardInterrupt:
        print("\nShutting down gracefully...")
    except Exception as e:
        print(f"Fatal error: {str(e)}")
    finally:
        await client.cleanup()

if __name__ == "__main__":
    asyncio.run(main())
