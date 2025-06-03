import asyncio
from mcp.server.fastmcp import FastMCP
import ollama # ollama.AsyncClient will be used
from pydantic import BaseModel, Field

# --- Pydantic models for input schemas ---
class BaseChatInput(BaseModel):
    user_message: str = Field(description="The message from the user.")
    model_name: str = Field(default="llama3.2", description="The Ollama model to use (e.g., 'llama3.2', 'mistral').")
    # Add llama3.2 instead of mistral as default, assuming you have it

# --- End Pydantic models ---

# 1. Initialize FastMCP server
mcp = FastMCP(name="DualChatbotPlatform_MCP")

# Initialize the Ollama async client (for streaming)
ollama_async_client = ollama.AsyncClient()

# 2. Define the base_chat MCP Tool (Streaming)
@mcp.tool(
    name="base_chat", # Renamed for clarity in our plan
    description="Sends a message to the base LLM and gets a streaming response.",
    input_schema=BaseChatInput
)
async def base_chat_tool(user_message: str, model_name: str = "llama3.2"): # Function is now async
    """
    Handles a chat request to the base LLM and streams the response.
    This will power the left-side (non-RAG) bot.
    """
    print(f"Received base_chat request for model '{model_name}': {user_message}")
    
    try:
        # Stream the response from Ollama using the async client
        async for part in await ollama_async_client.chat(
            model=model_name,
            messages=[{'role': 'user', 'content': user_message}],
            stream=True
        ):
            if part['message']['content']:
                chunk = part['message']['content']
                # FastMCP should handle yielding strings as text stream parts
                yield chunk 
            
            if part.get('done'):
                print(f"Base_chat stream finished for: {user_message[:30]}...")
                break
        
    except Exception as e:
        error_message = f"Error in base_chat_tool with Ollama: {e}"
        print(error_message)
        yield f"[ERROR: {error_message}]" # Stream the error back

@mcp.tool()
def list_ollama_models() -> str: # This tool can remain synchronous
    """Get a list of available Ollama models."""
    try:
        # Use the synchronous client for this simple, non-streaming task
        models_info = ollama.list() 
        model_names = [model['name'] for model in models_info['models']]
        if not model_names:
            return "No Ollama models found. Make sure Ollama is running and models are pulled."
        return f"Available models: {', '.join(model_names)}"
    except Exception as e:
        # Catch potential connection errors if Ollama isn't running
        return f"Error listing Ollama models: {str(e)}. Is Ollama running?"

# We will build the RAG chat tool and dual display logic later.
# For now, the 'dual_model_chat' you had can be kept for experimentation
# or removed if focusing only on the left/right bot plan.
# Let's keep it for now as it doesn't interfere.
@mcp.tool()
def dual_model_chat(message: str, model1: str = "llama3.2", model2: str = "llama3.2") -> str:
    """Get responses from two different models and compare them (non-streaming)."""
    try:
        response1 = ollama.chat(model=model1, messages=[{'role': 'user', 'content': message}])
        response2 = ollama.chat(model=model2, messages=[{'role': 'user', 'content': message}])
        
        result = f"Model {model1} says:\n{response1['message']['content']}\n\n"
        result += f"Model {model2} says:\n{response2['message']['content']}"
        return result
    except Exception as e:
        return f"Error in dual_model_chat: {str(e)}"

# 3. Main execution block
if __name__ == "__main__":
    print("Starting MCP Server for Dual Chatbot Platform...")
    try:
        # mcp.run() for FastMCP is synchronous at the top level,
        # but it can serve async tools.
        mcp.run(transport='stdio')
    except KeyboardInterrupt:
        print("\nMCP Server shutting down...")
    except Exception as e:
        print(f"An error occurred during server run: {e}")
    finally:
        print("MCP Server stopped.")