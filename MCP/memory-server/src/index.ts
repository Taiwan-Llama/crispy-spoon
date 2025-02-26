#!/usr/bin/env node

import { Server } from "@modelcontextprotocol/sdk/server/index.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import {
  CallToolRequestSchema,
  ListResourcesRequestSchema,
  ListToolsRequestSchema,
  ReadResourceRequestSchema,
  ErrorCode,
  McpError,
} from "@modelcontextprotocol/sdk/types.js";
import Graphology from "graphology";
import { ChromaClient } from "chromadb";
import axios from "axios";
import { format } from "date-fns";

// Types for ChromaDB interaction
interface ChromaCollection {
  add(ids: string[], embeddings: number[][], metadatas?: Record<string, any>[]): Promise<void>;
  query(queryEmbeddings: number[], n_results?: number): Promise<{ ids: string[][]; distances: number[][] }>;
  delete(ids: { ids: string[] }): Promise<void>;
}

// Types for our memory system
interface Memory {
  id: string;
  content: string;
  timestamp: string;
  type: "user_input" | "model_response";
}

interface Entity {
  id: string;
  name: string;
  type: string;
  observations: Array<{
    content: string;
    timestamp: string;
  }>;
}

interface Relationship {
  from: string;
  to: string;
  type: string;
  timestamp: string;
}

class MemoryServer {
  private server: Server;
  private graph: any; // Will store the graphology instance
  private vectorDb: ChromaClient | null = null;
  private memoryCollection: ChromaCollection | null = null;
  private memories: { [id: string]: Memory } = {};
  private entities: { [id: string]: Entity } = {};
  private relationships: Relationship[] = [];

  constructor() {
    this.server = new Server(
      {
        name: "memory-server",
        version: "0.1.0",
      },
      {
        capabilities: {
          resources: {},
          tools: {},
        },
      }
    );

    // Initialize graph as basic object with required methods
    this.graph = {
      addNode: (id: string, attrs: any) => {},
      addEdge: (from: string, to: string, attrs: any) => {},
      dropNode: (id: string) => {}
    };

    // Setup tools and resources
    this.setupTools();
    this.setupResources();
    
    // Initialize async components
    this.initialize().catch(console.error);

    // Error handling
    this.server.onerror = (error) => console.error("[MCP Error]", error);
    process.on("SIGINT", async () => {
      await this.server.close();
      process.exit(0);
    });
  }

  private async initialize() {
    try {
      // Initialize ChromaDB
      this.vectorDb = new ChromaClient();
      const collection = await this.vectorDb.createCollection({
        name: "memories",
        metadata: { description: "Memory storage for conversations" }
      });
      this.memoryCollection = collection as unknown as ChromaCollection;
    } catch (error) {
      console.error("ChromaDB initialization failed (vector search disabled):", error);
      this.vectorDb = null;
      this.memoryCollection = null;
    }
  }

  private setupTools() {
    this.server.setRequestHandler(ListToolsRequestSchema, async () => ({
      tools: [
        {
          name: "store_memory",
          description: "Store a new memory (user input or model response)",
          inputSchema: {
            type: "object",
            properties: {
              content: { type: "string", description: "Content of the memory" },
              type: { 
                type: "string", 
                enum: ["user_input", "model_response"],
                description: "Type of memory"
              }
            },
            required: ["content", "type"]
          }
        },
        {
          name: "create_entity",
          description: "Create a new entity in the knowledge graph",
          inputSchema: {
            type: "object",
            properties: {
              name: { type: "string", description: "Name of the entity" },
              type: { type: "string", description: "Type of entity" },
              observation: { type: "string", description: "Initial observation about the entity" }
            },
            required: ["name", "type", "observation"]
          }
        },
        {
          name: "add_relationship",
          description: "Create a relationship between entities",
          inputSchema: {
            type: "object",
            properties: {
              fromEntity: { type: "string", description: "Source entity ID" },
              toEntity: { type: "string", description: "Target entity ID" },
              type: { type: "string", description: "Type of relationship" }
            },
            required: ["fromEntity", "toEntity", "type"]
          }
        }
      ]
    }));

    this.server.setRequestHandler(CallToolRequestSchema, async (request) => {
      const timestamp = format(new Date(), "yyyy-MM-dd'T'HH:mm:ssXXX");

      switch (request.params.name) {
        case "store_memory": {
          const { content, type } = request.params.arguments as { content: string; type: string };
          const id = `mem_${Date.now()}`;
          
          const memory: Memory = {
            id,
            content,
            timestamp,
            type: type as "user_input" | "model_response"
          };

          this.memories[id] = memory;
          
          if (this.memoryCollection) {
            const embedding = await this.getEmbedding(content);
            await this.memoryCollection.add([id], [embedding], [{ timestamp, type }]);
          }

          return {
            content: [{
              type: "text",
              text: `Stored memory ${id}`
            }]
          };
        }

        case "create_entity": {
          const { name, type, observation } = request.params.arguments as { 
            name: string;
            type: string;
            observation: string;
          };

          const id = `ent_${Date.now()}`;
          const entity: Entity = {
            id,
            name,
            type,
            observations: [{
              content: observation,
              timestamp
            }]
          };

          this.entities[id] = entity;
          this.graph.addNode(id, { name, type });

          return {
            content: [{
              type: "text",
              text: `Created entity ${id}`
            }]
          };
        }

        case "add_relationship": {
          const { fromEntity, toEntity, type } = request.params.arguments as {
            fromEntity: string;
            toEntity: string;
            type: string;
          };

          if (!this.entities[fromEntity] || !this.entities[toEntity]) {
            throw new McpError(ErrorCode.InvalidParams, "Entity not found");
          }

          const relationship: Relationship = {
            from: fromEntity,
            to: toEntity,
            type,
            timestamp
          };

          this.relationships.push(relationship);
          this.graph.addEdge(fromEntity, toEntity, { type });

          return {
            content: [{
              type: "text",
              text: `Added relationship: ${fromEntity} -[${type}]-> ${toEntity}`
            }]
          };
        }

        default:
          throw new McpError(ErrorCode.InvalidParams, "Unknown tool");
      }
    });
  }

  private setupResources() {
    this.server.setRequestHandler(ListResourcesRequestSchema, async () => ({
      resources: [
        ...Object.values(this.memories).map(memory => ({
          uri: `memory://${memory.id}`,
          name: `Memory ${memory.id}`,
          description: `${memory.type} from ${memory.timestamp}`,
          mimeType: "application/json"
        })),
        ...Object.values(this.entities).map(entity => ({
          uri: `entity://${entity.id}`,
          name: entity.name,
          description: `${entity.type} with ${entity.observations.length} observations`,
          mimeType: "application/json"
        }))
      ]
    }));

    this.server.setRequestHandler(ReadResourceRequestSchema, async (request) => {
      const url = new URL(request.params.uri);
      const id = url.pathname.slice(2);

      if (url.protocol === "memory:") {
        const memory = this.memories[id];
        if (!memory) {
          throw new McpError(ErrorCode.InvalidParams, "Memory not found");
        }

        return {
          contents: [{
            uri: request.params.uri,
            mimeType: "application/json",
            text: JSON.stringify(memory, null, 2)
          }]
        };
      }

      if (url.protocol === "entity:") {
        const entity = this.entities[id];
        if (!entity) {
          throw new McpError(ErrorCode.InvalidParams, "Entity not found");
        }

        return {
          contents: [{
            uri: request.params.uri,
            mimeType: "application/json",
            text: JSON.stringify(entity, null, 2)
          }]
        };
      }

      throw new McpError(ErrorCode.InvalidParams, "Invalid resource URI");
    });
  }

  private async getEmbedding(text: string): Promise<number[]> {
    try {
      const model = process.env.OLLAMA_MODEL || "llama2";
      const response = await axios.post("http://localhost:11434/api/embeddings", {
        model,
        prompt: text
      });
      return response.data.embedding;
    } catch (error) {
      console.error("Error getting embedding:", error);
      return new Array(384).fill(0);
    }
  }

  async run() {
    const transport = new StdioServerTransport();
    await this.server.connect(transport);
    console.error("Memory MCP server running on stdio");
  }
}

const server = new MemoryServer();
server.run().catch((error) => {
  console.error("Server error:", error);
  process.exit(1);
});
