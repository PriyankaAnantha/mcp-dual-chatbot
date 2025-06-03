import asyncio
from mcp.server.fastmcp import FastMCP
import ollama
from pydantic import BaseModel, Field
import time

# --- Pydantic models for input schemas ---
class BaseChatInput(BaseModel):
    user_message: str = Field(description="The message from the user.")
    model_name: str = Field(default="mistral", description="The Ollama model to use (e.g., 'mistral', 'llama3').")
# --- End Pydantic models ---

# 1. Initialize FastMCP server
mcp = FastMCP(name="DualChatbotPlatform_MCP")

try:
    ollama_async_client = ollama.AsyncClient()
except Exception as e:
    print(f"WARNING: Could not initialize Ollama AsyncClient. Ensure Ollama is running. Error: {e}")
    ollama_async_client = None

# 2. Define the base_chat MCP Tool (Streaming)
@mcp.tool(
    name="base_chat",
    description="Sends a message to a selected LLM and gets a response."
)

async def base_chat_tool(user_message: str, model_name: str = BaseChatInput.model_fields['model_name'].default): 
    start_time = time.time()
    print(f"MCP_SERVER: [{start_time:.2f}] Received base_chat for model '{model_name}': '{user_message[:50]}...'")
    
    if not ollama_async_client:
        error_msg = "Ollama client not initialized."
        print(f"MCP_SERVER: [{time.time() - start_time:.2f}s] ERROR - {error_msg}")
        return f"[ERROR: {error_msg}]"

    full_response_content = []
    try:
        print(f"MCP_SERVER: [{time.time() - start_time:.2f}s] Calling Ollama model '{model_name}'...")
        ollama_call_start_time = time.time()
        
        first_chunk_received_time = None

        async for part in await ollama_async_client.chat(
            model=model_name,
            messages=[{'role': 'user', 'content': user_message}],
            stream=True
        ):
            if first_chunk_received_time is None:
                first_chunk_received_time = time.time()
                print(f"MCP_SERVER: [{first_chunk_received_time - start_time:.2f}s] First chunk from Ollama after {first_chunk_received_time - ollama_call_start_time:.2f}s.")

            if part.get('message', {}).get('content'):
                chunk = part['message']['content']
                full_response_content.append(chunk)
            
            if part.get('done'):
                if part.get('done_reason') == 'load_model':
                    print(f"MCP_SERVER: [{time.time() - start_time:.2f}s] Ollama is loading model '{model_name}'...")
                else:
                    print(f"MCP_SERVER: [{time.time() - start_time:.2f}s] Base_chat Ollama stream finished.")
                    break
        
        ollama_call_duration = time.time() - ollama_call_start_time
        print(f"MCP_SERVER: [{time.time() - start_time:.2f}s] Ollama call took {ollama_call_duration:.2f}s.")
        
        final_response = "".join(full_response_content)
        # print(f"MCP_SERVER: [{time.time() - start_time:.2f}s] Final collected response: {final_response[:100]}...")
        
        total_tool_duration = time.time() - start_time
        print(f"MCP_SERVER: [{total_tool_duration:.2f}s] Returning final response. Length: {len(final_response)}")
        return final_response 
        
    except ollama.ResponseError as e:
        # ... (error handling) ...
        error_message = f"Ollama API error: {e.error} (Status: {e.status_code})"
        print(f"MCP_SERVER: [{time.time() - start_time:.2f}s] ERROR - {error_message}")
        return f"[ERROR: {error_message}]"
    except Exception as e:
        error_message = f"Error in base_chat_tool with Ollama: {e}"
        print(f"MCP_SERVER: [{time.time() - start_time:.2f}s] ERROR - {error_message}")
        return f"[ERROR: {error_message}]"


# 3. Main execution block
if __name__ == "__main__":
    print("Starting MCP Server for Dual Chatbot Platform (FastMCP)...")
    try:
        mcp.run(transport='stdio')
    except KeyboardInterrupt:
        print("\nMCP Server shutting down...")
    except Exception as e:
        print(f"An error occurred during server run: {e}")
    finally:
        print("MCP Server stopped.")