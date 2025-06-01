# Project Plan: MCP Dual Chatbot Comparison Platform

**Overall Project Goal:** To develop a Python-based MCP server and a modern web interface. The platform will allow users to select an LLM, upload documents for context, and chat with two side-by-side bots: a base LLM (left) and an LLM augmented with the uploaded document context (right, "fine-tuned"). Users will be able to compare responses and save conversation history.

**Key Interpretation:**
*   The "fine-tuned bot" will primarily achieve its enhanced capabilities through Retrieval Augmented Generation (RAG), using the user-uploaded documents to provide context at inference time.
*   "Model selection" will allow choosing the base LLM used by *both* bots, with the right-side bot always having the additional RAG context.

---

## Phase 1: Core MCP Server & Base LLM Integration

### 1. Task: Project Setup and MCP Server Foundation
*   **Description:** Initialize the Python project (`uv` or `pip/venv`), install `model-context-protocol[cli]`, `FastAPI` (for web APIs), and a base LLM client library (e.g., `openai`, `anthropic`, or `ollama-python` for local models like Mistral, Qwen). Implement a minimal `FastMCP` server that can be started.
*   **Deliverable:** A runnable, basic MCP server script.

### 2. Task: Integrate First LLM and Basic Chat Tool (Streaming)
*   **Description:** Integrate one LLM (e.g., Mistral via Ollama). Create an MCP `@mcp.tool()` named `base_chat` that takes user input, interacts with the LLM, and streams the response. This tool will power the left-side (non-RAG) bot.
*   **Deliverable:** MCP server with a `base_chat` tool capable of streaming responses from the selected LLM. Testable via `mcp` CLI.

---

## Phase 2: RAG Implementation for the "Fine-Tuned" Bot

### 3. Task: Document Processing for RAG Context
*   **Description:** Implement an MCP tool (e.g., `process_document_for_rag`) that accepts a document (e.g., file path or content via API). This tool will handle text extraction (if needed), chunking, and embedding generation using a library like `sentence-transformers`. Store these embeddings (initially in-memory or a simple local vector store like FAISS/ChromaDB).
*   **Deliverable:** Server can process documents into queryable embeddings.

### 4. Task: RAG-Enhanced Chat Tool (Streaming)
*   **Description:** Create a new MCP `@mcp.tool()` named `rag_chat`. This tool will take user input, embed it, retrieve relevant context from the processed documents (vector store), construct a prompt including this context, interact with the *same selected base LLM*, and stream the response. This will power the right-side ("fine-tuned") bot.
*   **Deliverable:** MCP server with a `rag_chat` tool that provides context-aware, streaming responses.

---

## Phase 3: Web Application Backend & API Development

### 5. Task: Design and Implement FastAPI Backend APIs
*   **Description:** Using FastAPI, create API endpoints that the web frontend will use. These include:
    *   Endpoint to list available LLMs for selection.
    *   Endpoint for document upload (which will trigger the `process_document_for_rag` MCP tool).
    *   Endpoints to initiate and stream responses from `base_chat` and `rag_chat` MCP tools (potentially via WebSockets or Server-Sent Events for streaming to the web UI).
    *   Endpoints for saving and retrieving conversation history.
*   **Deliverable:** A set of FastAPI endpoints that expose the MCP server's functionalities to a web client.

### 6. Task: MCP Server as a Subprocess/Managed Service
*   **Description:** Ensure the FastAPI application can start and manage the MCP server (e.g., as a subprocess if using `stdio` transport for MCP). The FastAPI app will act as the primary interface for the web UI.
*   **Deliverable:** FastAPI backend successfully communicates with the MCP server.

---

## Phase 4: Web Interface Frontend Development

### 7. Task: Basic Web UI Structure and Styling
*   **Description:** Develop the basic HTML, CSS, and JavaScript structure for the web interface using a modern frontend framework (e.g., React, Vue, or plain JS with a library like HTMX). Focus on a clean, modern look and feel.
*   **Deliverable:** A static or partially dynamic web page layout for the dual chatbot interface, model selection, and document upload.

### 8. Task: Implement Dual Chatbot UI and Comparison View
*   **Description:** Create two side-by-side chat interfaces. Implement JavaScript logic to send user queries to the respective backend APIs (`base_chat` and `rag_chat`) and display the streamed responses in real-time in the correct chat windows. Ensure users can easily compare the outputs.
*   **Deliverable:** Functional dual chatbot UI where users can send a query and see responses from both bots.

### 9. Task: Implement Model Selection and Document Upload UI
*   **Description:** Add UI elements for selecting the base LLM (dropdown/radio buttons). Implement a file upload component that calls the document upload API. Provide feedback to the user on upload status and processing.
*   **Deliverable:** Users can select models and upload documents through the web interface.

---

## Phase 5: User Features & Finalization

### 10. Task: Conversation History Implementation (Frontend & Backend)
*   **Description:** Implement the UI for displaying conversation history. Ensure the backend API correctly saves new messages and retrieves history for a given user/session. (For user-specific history, a simple session ID or user input for a username can be used initially).
*   **Deliverable:** Users can see past interactions, and conversations are persisted.

### 11. Task: Multi-LLM Backend Switching
*   **Description:** Ensure the MCP server (and by extension, the FastAPI backend) can dynamically switch the underlying LLM for both `base_chat` and `rag_chat` tools based on user selection from the UI.
*   **Deliverable:** Model selection in the UI correctly changes the LLM used for generation.

### 12. Task: Final Testing, Refinement, and Deployment Prep
*   **Description:** Conduct thorough end-to-end testing of all features. Refine UI/UX based on testing. Prepare basic deployment instructions (e.g., running locally).
*   **Deliverable:** A polished, well-tested application and basic deployment documentation.

---