#!/usr/bin/env node
import { Server } from "@modelcontextprotocol/sdk/server/index.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import { CallToolRequestSchema, ListToolsRequestSchema, McpError, ErrorCode, } from "@modelcontextprotocol/sdk/types.js";
import axios from 'axios';
class DuckDuckGoSearchServer {
    server;
    searchEndpoint = 'https://html.duckduckgo.com/html/';
    constructor() {
        this.server = new Server({
            name: "duckduckgo-search-server",
            version: "0.1.0",
        }, {
            capabilities: {
                tools: {
                    schema: {
                        toolInputs: {
                            web_search: {
                                type: "object",
                                properties: {
                                    query: {
                                        type: "string",
                                        description: "Search query"
                                    },
                                    limit: {
                                        type: "number",
                                        description: "Maximum number of results (1-25)",
                                        minimum: 1,
                                        maximum: 25,
                                        default: 10
                                    }
                                },
                                required: ["query"]
                            },
                            summarize_search: {
                                type: "object",
                                properties: {
                                    query: {
                                        type: "string",
                                        description: "Search query"
                                    },
                                    limit: {
                                        type: "number",
                                        description: "Maximum number of results to summarize (1-10)",
                                        minimum: 1,
                                        maximum: 10,
                                        default: 5
                                    }
                                },
                                required: ["query"]
                            }
                        }
                    }
                }
            }
        });
        this.setupHandlers();
    }
    setupHandlers() {
        this.server.setRequestHandler(ListToolsRequestSchema, async () => ({
            tools: [
                {
                    name: "web_search",
                    description: "Search the web using DuckDuckGo",
                    inputSchema: {
                        type: "object",
                        properties: {
                            query: {
                                type: "string",
                                description: "Search query"
                            },
                            limit: {
                                type: "number",
                                description: "Maximum number of results (1-25)",
                                minimum: 1,
                                maximum: 25,
                                default: 10
                            }
                        },
                        required: ["query"]
                    }
                },
                {
                    name: "summarize_search",
                    description: "Search and summarize results",
                    inputSchema: {
                        type: "object",
                        properties: {
                            query: {
                                type: "string",
                                description: "Search query"
                            },
                            limit: {
                                type: "number",
                                description: "Maximum number of results to summarize (1-10)",
                                minimum: 1,
                                maximum: 10,
                                default: 5
                            }
                        },
                        required: ["query"]
                    }
                }
            ]
        }));
        this.server.setRequestHandler(CallToolRequestSchema, async (request) => {
            switch (request.params.name) {
                case "web_search":
                    return this.handleWebSearch(request);
                case "summarize_search":
                    return this.handleSummarizeSearch(request);
                default:
                    throw new McpError(ErrorCode.MethodNotFound, "Unknown tool");
            }
        });
    }
    async handleWebSearch(request) {
        const query = String(request.params.arguments?.query);
        const limit = Math.min(Number(request.params.arguments?.limit) || 10, 25);
        if (!query) {
            throw new McpError(ErrorCode.InvalidParams, "Query is required");
        }
        try {
            const results = await this.performSearch(query, limit);
            return {
                content: [{
                        type: "text",
                        text: JSON.stringify(results, null, 2)
                    }]
            };
        }
        catch (error) {
            const message = error instanceof Error ? error.message : String(error);
            throw new McpError(ErrorCode.InternalError, `Search failed: ${message}`);
        }
    }
    async handleSummarizeSearch(request) {
        const query = String(request.params.arguments?.query);
        const limit = Math.min(Number(request.params.arguments?.limit) || 5, 10);
        if (!query) {
            throw new McpError(ErrorCode.InvalidParams, "Query is required");
        }
        try {
            const results = await this.performSearch(query, limit);
            const summary = this.generateSearchSummary(results);
            return {
                content: [{
                        type: "text",
                        text: summary
                    }]
            };
        }
        catch (error) {
            const message = error instanceof Error ? error.message : String(error);
            throw new McpError(ErrorCode.InternalError, `Summary failed: ${message}`);
        }
    }
    async performSearch(query, limit) {
        try {
            console.error('Sending request to DuckDuckGo...');
            const response = await axios.post(this.searchEndpoint, new URLSearchParams({
                q: query,
                s: '0'
            }).toString(), {
                headers: {
                    'Content-Type': 'application/x-www-form-urlencoded',
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
                }
            });
            console.error('Got response from DuckDuckGo');
            const results = [];
            // Use regex to parse results from HTML
            const resultRegex = /<h2 class="result__title">.*?<a rel="nofollow" class="result__a" href="([^"]+)".*?>([^<]+)<\/a>.*?<a class="result__snippet".*?>([^<]+)<\/a>/gs;
            let match;
            const html = response.data;
            while ((match = resultRegex.exec(html)) !== null && results.length < limit) {
                const [, link, title, snippet] = match;
                if (link && title) {
                    results.push({
                        title: title.trim(),
                        link: link,
                        snippet: snippet ? snippet.trim() : ''
                    });
                }
            }
            return results;
        }
        catch (error) {
            console.error('Search error:', error);
            throw new Error('Failed to perform search');
        }
    }
    generateSearchSummary(results) {
        if (results.length === 0) {
            return "No results found.";
        }
        const summary = results.map(result => {
            return `📌 ${result.title}\n${result.snippet}\nSource: ${result.link}\n`;
        }).join('\n');
        return `Found ${results.length} results:\n\n${summary}`;
    }
    async start() {
        try {
            const transport = new StdioServerTransport();
            await this.server.connect(transport);
            console.error('DuckDuckGo search server running on stdio');
        }
        catch (error) {
            console.error('Server error:', error);
            process.exit(1);
        }
    }
}
const server = new DuckDuckGoSearchServer();
server.start();
