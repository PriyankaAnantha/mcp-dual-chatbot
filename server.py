import asyncio
from mcp.server.fastmcp import FastMCP
import ollama
import logging

# Minimal logging
logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger("DualChatbotMCP")

# Initialize FastMCP server
mcp = FastMCP(name="DualChatbotPlatform_MCP")

# Global Ollama client
ollama_client = None
ollama_available = False

async def check_ollama():
    """Check if Ollama is available and working"""
    global ollama_client, ollama_available
    
    try:
        # Create client
        ollama_client = ollama.AsyncClient()
        
        # Quick test - just try to list models with short timeout
        models = await asyncio.wait_for(ollama_client.list(), timeout=3.0)
        ollama_available = True
        return True
    except:
        ollama_available = False
        return False

@mcp.tool(
    name="test_connection",
    description="Test if the MCP server is working properly"
)
async def test_connection():
    """Simple test tool to verify MCP is working"""
    return "MCP server is working! Connection successful."

@mcp.tool(
    name="list_models", 
    description="List available Ollama models"
)
async def list_models():
    """List available Ollama models"""
    if not ollama_available:
        if not await check_ollama():
            return "ERROR: Ollama not available. Is it running on localhost:11434?"
    
    try:
        models_response = await asyncio.wait_for(ollama_client.list(), timeout=5.0)
        
        # Handle Ollama's custom response type
        if hasattr(models_response, 'models'):
            models_list = models_response.models
        else:
            # Fallback to dict-like access
            models_list = getattr(models_response, 'get', lambda k, default: default)('models', [])
        
        if not models_list:
            return "No models found. Try 'ollama pull mistral' to download a model."
        
        # Extract model names
        model_names = []
        for model in models_list:
            if hasattr(model, 'name'):
                model_names.append(model.name)
            elif hasattr(model, 'model'):
                model_names.append(model.model)
            elif isinstance(model, dict):
                model_names.append(model.get('name', model.get('model', str(model))))
            else:
                model_names.append(str(model))
        
        if model_names:
            return f"Available models: {', '.join(model_names)}"
        else:
            return f"Models found but could not extract names. Count: {len(models_list)}"
            
    except asyncio.TimeoutError:
        return "ERROR: Timeout connecting to Ollama"
    except Exception as e:
        return f"ERROR: {type(e).__name__}: {str(e)}"

@mcp.tool(
    name="base_chat",
    description="Chat with a base LLM model"
)
async def base_chat(user_message: str, model_name: str = "mistral"):
    """Chat with base LLM without document context"""
    
    # Check if Ollama is available
    if not ollama_available:
        if not await check_ollama():
            return "ERROR: Ollama not available. Please start Ollama service first."
    
    # Validate inputs
    if not user_message or not user_message.strip():
        return "ERROR: Please provide a message"
    
    try:
        # Collect response parts
        response_parts = []
        
        # Stream response from Ollama
        stream = await asyncio.wait_for(
            ollama_client.chat(
                model=model_name,
                messages=[{'role': 'user', 'content': user_message.strip()}],
                stream=True
            ),
            timeout=5.0  # Timeout for getting the stream
        )
        
        # Process stream with overall timeout
        async def process_stream():
            async for chunk in stream:
                if chunk.get('message', {}).get('content'):
                    response_parts.append(chunk['message']['content'])
                if chunk.get('done', False):
                    break
            return ''.join(response_parts)
        
        result = await asyncio.wait_for(process_stream(), timeout=25.0)
        
        if not result:
            return "No response generated from model"
        
        return result
        
    except asyncio.TimeoutError:
        return f"ERROR: Timeout - model '{model_name}' took too long to respond"
    except ollama.ResponseError as e:
        if "not found" in str(e).lower():
            return f"ERROR: Model '{model_name}' not found. Try: ollama pull {model_name}"
        return f"ERROR: Ollama error - {str(e)}"
    except Exception as e:
        return f"ERROR: {type(e).__name__} - {str(e)}"

# Run the server
if __name__ == "__main__":
    # Initialize Ollama check on startup
    asyncio.create_task(check_ollama())
    
    # Run MCP server
    mcp.run(transport='stdio')