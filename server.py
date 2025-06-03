import asyncio
import time
from mcp.server.fastmcp import FastMCP
import ollama
from pydantic import BaseModel, Field
import logging # Import logging

# --- Configure logging ---
# Get the root logger for mcp (or configure your own)
# mcp dev might already configure a handler that shows up in its "Error output from MCP server"
# For local debugging, let's ensure our logs go to stderr so mcp dev's proxy ignores them for protocol.
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("DualChatbotMCP") # Use a specific name for your logger

# --- Pydantic models for input schemas ---
class BaseChatInput(BaseModel):
    user_message: str = Field(description="The message from the user.")
    model_name: str = Field(default="mistral", description="The Ollama model to use (e.g., 'mistral', 'llama3').")

# 1. Initialize FastMCP server
mcp = FastMCP(name="DualChatbotPlatform_MCP")

try:
    ollama_async_client = ollama.AsyncClient()
    try:
        ollama.list() 
        logger.info("Successfully queried Ollama models on startup.") # USE LOGGER
    except Exception as e_init_list:
        logger.warning(f"Ollama service might not be responsive on startup: {e_init_list}") # USE LOGGER
except Exception as e_client:
    logger.critical(f"Could not initialize Ollama AsyncClient: {e_client}") # USE LOGGER
    ollama_async_client = None

@mcp.tool(
    name="base_chat",
    description="Sends a message to a selected LLM and gets a full response (modified to be non-streaming to client for timeout debug)."
)
async def base_chat_tool(user_message: str, model_name: str = BaseChatInput.model_fields['model_name'].default): 
    func_start_time_monotonic = time.monotonic()
    logger.info(f"[T+{func_start_time_monotonic:.2f}s] base_chat_tool started. Model: '{model_name}', Message: '{user_message[:50]}...'")
    
    if not ollama_async_client:
        error_msg = "Ollama async client not initialized. Check server startup logs."
        logger.error(f"[T+{time.monotonic() - func_start_time_monotonic:.2f}s] {error_msg}")
        # For a non-streaming tool, return the error. FastMCP wraps strings.
        return f"[ERROR: {error_msg}]" 

    full_response_content = []
    try:
        logger.info(f"[T+{time.monotonic() - func_start_time_monotonic:.2f}s] Calling ollama_async_client.chat for model '{model_name}'...")
        ollama_api_call_start_monotonic = time.monotonic()
        
        first_chunk_received_monotonic = None
        stream_done_monotonic = None

        async for part in await ollama_async_client.chat(
            model=model_name,
            messages=[{'role': 'user', 'content': user_message}],
            stream=True 
        ):
            if first_chunk_received_monotonic is None:
                first_chunk_received_monotonic = time.monotonic()
                logger.info(f"[T+{first_chunk_received_monotonic - func_start_time_monotonic:.2f}s] First chunk from Ollama after {first_chunk_received_monotonic - ollama_api_call_start_monotonic:.2f}s of API call.")

            if part.get('message', {}).get('content'):
                chunk = part['message']['content']
                full_response_content.append(chunk)
            
            if part.get('done'):
                stream_done_monotonic = time.monotonic()
                if part.get('done_reason') == 'load_model':
                    logger.info(f"[T+{stream_done_monotonic - func_start_time_monotonic:.2f}s] Ollama is loading model '{model_name}' (API call {stream_done_monotonic - ollama_api_call_start_monotonic:.2f}s so far).")
                else:
                    logger.info(f"[T+{stream_done_monotonic - func_start_time_monotonic:.2f}s] Ollama stream 'done'. API call part took {stream_done_monotonic - ollama_api_call_start_monotonic:.2f}s.")
                    break
        
        if stream_done_monotonic is None:
            stream_done_monotonic = time.monotonic()
            logger.warning(f"[T+{stream_done_monotonic - func_start_time_monotonic:.2f}s] Ollama stream appears to have ended without a 'done' flag. API call part took {stream_done_monotonic - ollama_api_call_start_monotonic:.2f}s.")

        final_response_str = "".join(full_response_content)
        
        func_end_time_monotonic = time.monotonic()
        total_tool_duration = func_end_time_monotonic - func_start_time_monotonic
        logger.info(f"[T+{total_tool_duration:.2f}s] Returning final response. Length: {len(final_response_str)}. Full duration: {total_tool_duration:.2f}s")
        return final_response_str
        
    except ollama.ResponseError as e:
        error_end_time = time.monotonic()
        error_message = f"Ollama API error: {e.error} (Status: {e.status_code})"
        if hasattr(e, 'error') and e.error and "model" in str(e.error).lower() and "not found" in str(e.error).lower():
            error_message = f"Ollama model '{model_name}' not found. Try 'ollama pull {model_name}'."
        logger.error(f"[T+{error_end_time - func_start_time_monotonic:.2f}s] Ollama ResponseError: {error_message}", exc_info=True)
        return f"[ERROR: {error_message}]"
    except Exception as e:
        error_end_time = time.monotonic()
        error_message = f"Error in base_chat_tool with Ollama: {type(e).__name__} - {e}"
        logger.error(f"[T+{error_end_time - func_start_time_monotonic:.2f}s] Exception: {error_message}", exc_info=True)
        return f"[ERROR: {error_message}]"

# Main execution block
if __name__ == "__main__":
    logger.info("Starting MCP Server for Dual Chatbot Platform (FastMCP)...") # USE LOGGER
    try:
        sync_ollama_models = ollama.list()
        if not sync_ollama_models.get('models'):
            logger.warning("No models found in Ollama. Ensure models are pulled via 'ollama pull <modelname>'.") # USE LOGGER
        else:
            logger.info(f"Ollama models available: {[m['name'] for m in sync_ollama_models['models']]}") # USE LOGGER
    except Exception as e_main:
        logger.critical(f"Cannot connect to Ollama service during startup check: {e_main}") # USE LOGGER
        logger.critical("Please ensure Ollama is installed, running, and accessible at http://localhost:11434.")
    
    try:
        mcp.run(transport='stdio')
    except KeyboardInterrupt:
        logger.info("\nMCP Server shutting down...") # USE LOGGER
    except Exception as e_run:
        logger.error(f"An error occurred during server run: {e_run}", exc_info=True) # USE LOGGER
    finally:
        logger.info("MCP Server stopped.") # USE LOGGER