# DC_MCP_BOT 簡介

DC_MCP_BOT 是一款基於 Discord 的智能機器人，結合了 **Model Context Protocol (MCP)**、**LlamaIndex** 文檔檢索功能，以及 **ChromaDB** 記憶管理系統，提供高效的對話與資訊檢索能力。透過整合來自多個來源的強大工具，DC_MCP_BOT 能夠在 Discord 伺服器中執行複雜的查詢、回應用戶問題，並記錄與檢索過往的對話內容。

---

## 主要功能

1. **MCP 工具集成**  
   - **瀏覽器自動化 (Puppeteer)**：讓機器人能夠透過 MCP 介面操作網站，擷取網頁資訊。  
   - **記憶系統 (Memory Server)**：記錄對話歷史，幫助機器人理解上下文，提供更個性化的回應。  
   - **DuckDuckGo 搜尋**：透過 DuckDuckGo 搜尋引擎獲取最新資訊。  
   - **文件系統操作 (Filesystem Server)**：支援存取及查詢本地文件。  
   - **邏輯推理 (Sequential Thinking)**：讓機器人能夠處理複雜問題並拆解成步驟回答。

2. **文檔檢索與管理**  
   - 支援上傳 `.txt`、`.py`、`.md`、`.json` 等格式的文件，並自動建立嵌入向量索引。  
   - 可透過 `!query <問題>` 指令搜尋儲存的文件，獲取最相關的資訊。

3. **記憶與對話記錄**  
   - 採用 **ChromaDB** 來存儲和檢索對話記錄，確保機器人能根據歷史對話提供更精確的回應。  
   - 可手動保存對話 (`!save` 指令)，以便未來查閱。

4. **高級 AI 推理與對話**  
   - 內建 **OpenAI API**，使用強大的 LLM (如 Mistral) 進行自然語言理解與回應生成。  
   - 結合 MCP 工具與文件檢索技術，提升回答的準確性與可用性。  

---

## 使用方法

1. **安裝與啟動**  
   - 確保 `.env` 文件內設置了 `DISCORD_TOKEN`，然後執行 `python DC_MCP_BOT.py` 啟動機器人。

2. **基本指令**
   - `!save`：保存當前的對話記錄。
   - `!query <問題>`：查詢上傳的文件以獲取相關資訊。
   - `!servers`：查看當前可用的 MCP 伺服器狀態。

3. **文件管理**
   - 上傳 `.txt`、`.md` 等格式的文件，機器人會自動處理並建立索引，可供後續查詢。

4. **智慧對話**
   - 直接在 Discord 頻道輸入問題，機器人將根據上下文和過往記錄提供最佳回應。

---
## 存在的問題
1. **記憶的時間不準確**
   -沒辦法準確記住上一輪的對話，導致回答不準確。

2. **檔案上傳的限制**
   -檔案上傳的大小限制為8MB，超過的檔案無法上傳。(DC的限制)

3. **檔案上傳的格式限制**
   -只能上傳`.txt`、`.md`、`.py`、`.json`等格式的檔案，其他格式的檔案無法上傳。

4. **LLM對MCP伺服器的操作問題**
   -LLM對MCP的操作不穩定，導致無法正常使用MCP伺服器的功能。

---
## 特別鳴謝

DC_MCP_BOT 的開發得到了以下開源專案的啟發與支援：

- **[Model Context Protocol (MCP)](https://github.com/modelcontextprotocol/servers)**  
  提供標準化的工具協議，使機器人能夠輕鬆調用各種外部服務。

- **[sthenno](https://huggingface.co/sthenno) 的系統提示靈感**  
  參考其提示設計，使機器人能夠以更流暢的方式進行回應與交互。

- **[LlamaIndex](https://gpt-index.readthedocs.io/)**  
  強大的文件管理與檢索功能，使機器人能夠高效處理用戶上傳的文檔。

- **DC 平台**  
  為機器人提供了運行環境與測試支持。

感謝這些專案的貢獻，使 DC_MCP_BOT 能夠更強大、靈活且實用！