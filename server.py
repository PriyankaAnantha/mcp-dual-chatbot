import asyncio
from mcp.server.fastmcp import FastMCP
import ollama
import logging
import json
from typing import AsyncGenerator, Optional

# Minimal logging to avoid MCP protocol interference
logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger("DualChatbotMCP")

# Initialize FastMCP server
mcp = FastMCP(name="DualChatbotPlatform_MCP")

# Global Ollama client state
ollama_client = None
ollama_available = False
default_model = "mistral"

async def check_ollama():
    """Check if Ollama is available and initialize client"""
    global ollama_client, ollama_available
    
    try:
        ollama_client = ollama.AsyncClient()
        # Test connection with a quick model list call
        await asyncio.wait_for(ollama_client.list(), timeout=5.0)
        ollama_available = True
        logger.info("✅ Ollama connection established")
        return True
    except asyncio.TimeoutError:
        logger.error("❌ Ollama connection timeout - service may be slow")
        ollama_available = False
        return False
    except Exception as e:
        logger.error(f"❌ Ollama connection failed: {e}")
        ollama_available = False
        return False

async def ensure_ollama_available():
    """Ensure Ollama is available, attempt reconnection if needed"""
    if not ollama_available:
        return await check_ollama()
    return True

@mcp.tool(
    name="test_connection",
    description="Test if the MCP server is working properly"
)
async def test_connection():
    """Simple test tool to verify MCP server connectivity"""
    status = await check_ollama()
    if status:
        return "✅ MCP server is working! Ollama connection successful."
    else:
        return "⚠️ MCP server working, but Ollama connection failed. Is Ollama running?"

@mcp.tool(
    name="list_models", 
    description="List all available Ollama models for selection"
)
async def list_models():
    """List all available Ollama models that can be used for chat"""
    if not await ensure_ollama_available():
        return {
            "status": "error",
            "message": "❌ ERROR: Ollama not available. Please ensure Ollama service is running:\n" +
                      "- Windows: Start Ollama desktop app or run 'ollama serve'\n" +
                      "- Linux: Run 'sudo systemctl start ollama' or 'ollama serve'\n" +
                      "- Test with: 'ollama list'"
        }
    
    try:
        models_response = await asyncio.wait_for(ollama_client.list(), timeout=10.0)
        
        # Handle Ollama's custom ListResponse type
        if hasattr(models_response, 'models'):
            models_list = models_response.models
        else:
            models_list = []
        
        if not models_list:
            return {
                "status": "warning",
                "message": "❌ No models found. Download a model first:\n" +
                          "Example: ollama pull mistral\n" +
                          "Or: ollama pull llama3.2"
            }
        
        # Extract model names and details
        available_models = []
        for model in models_list:
            if hasattr(model, 'name'):
                available_models.append({
                    "name": model.name,
                    "size": getattr(model, 'size', 'Unknown'),
                    "modified": str(getattr(model, 'modified_at', 'Unknown'))
                })
            else:
                available_models.append({"name": str(model), "size": "Unknown"})
        
        return {
            "status": "success",
            "models": available_models,
            "count": len(available_models),
            "message": f"📋 Found {len(available_models)} available models"
        }
            
    except asyncio.TimeoutError:
        return {
            "status": "error",
            "message": "⏱️ ERROR: Timeout connecting to Ollama service (>10s)"
        }
    except Exception as e:
        return {
            "status": "error", 
            "message": f"❌ ERROR: {type(e).__name__}: {str(e)}"
        }

@mcp.tool(
    name="base_chat",
    description="Chat with a base LLM model (left-side bot for comparison platform). Supports streaming responses."
)
async def base_chat(user_message: str, model_name: str = None, stream: bool = True, max_tokens: int = 1000):
    """
    Enhanced chat function with streaming support for Task 2 - Phase 1
    This powers the left-side (base) chatbot in the dual comparison interface.
    """
    
    if not user_message or not user_message.strip():
        return {
            "status": "error",
            "message": "❌ ERROR: No message provided"
        }
    
    # Use default model if none specified
    if not model_name:
        model_name = default_model
    
    if not await ensure_ollama_available():
        return {
            "status": "error",
            "message": "❌ ERROR: Ollama not available. Please start Ollama service first."
        }
    
    try:
        # Prepare the chat parameters
        chat_params = {
            "model": model_name,
            "messages": [{"role": "user", "content": user_message.strip()}],
            "stream": stream
        }
        
        if stream:
            # Streaming response - collect all chunks
            response_chunks = []
            full_response = ""
            
            # Add longer timeout for streaming to handle model loading
            try:
                # The ollama client.chat() with stream=True returns an async generator directly
                stream_gen = ollama_client.chat(**chat_params)
                
                # Add timeout wrapper for the entire streaming process
                async def stream_with_timeout():
                    async for chunk in stream_gen:
                        if isinstance(chunk, dict) and 'message' in chunk:
                            content = chunk['message'].get('content', '')
                        elif hasattr(chunk, 'message') and hasattr(chunk.message, 'content'):
                            content = chunk.message.content
                        else:
                            content = ""
                            
                        if content:
                            response_chunks.append(content)
                            full_response += content
                
                # Give streaming more time for initial model load
                await asyncio.wait_for(stream_with_timeout(), timeout=60.0)
                
            except asyncio.TimeoutError:
                return {
                    "status": "error",
                    "message": f"⏱️ STREAMING TIMEOUT: {model_name} took >60s - model may be loading for first time",
                    "model_used": model_name,
                    "suggestion": "Try 'preload_model' tool first, or use a smaller model"
                }
            
            if not full_response:
                return {
                    "status": "warning",
                    "message": "⚠️ Model responded but with empty content",
                    "model_used": model_name
                }
            
            return {
                "status": "success",
                "response": full_response,
                "model_used": model_name,
                "streaming": True,
                "chunks_received": len(response_chunks),
                "message": f"✅ Streaming response completed from {model_name}"
            }
        
        else:
            # Non-streaming response with longer timeout for model loading
            response = await asyncio.wait_for(
                ollama_client.chat(**chat_params),
                timeout=60.0  # Increased from 30s to 60s
            )
            
            # Extract response content
            if hasattr(response, 'message') and hasattr(response.message, 'content'):
                content = response.message.content
                if content:
                    return {
                        "status": "success",
                        "response": content,
                        "model_used": model_name,
                        "streaming": False,
                        "message": f"✅ Response received from {model_name}"
                    }
            
            return {
                "status": "error",
                "message": f"❌ Could not parse response from {model_name}",
                "model_used": model_name
            }
        
    except asyncio.TimeoutError:
        return {
            "status": "error",
            "message": f"⏱️ TIMEOUT: {model_name} took >30s to respond - too slow for real-time chat",
            "model_used": model_name
        }
    except Exception as e:
        error_msg = str(e)
        if "model" in error_msg.lower() and "not found" in error_msg.lower():
            return {
                "status": "error",
                "message": f"❌ Model '{model_name}' not found. Available models: Use 'list_models' tool to see options.",
                "model_used": model_name
            }
        else:
            return {
                "status": "error",
                "message": f"❌ Chat error with {model_name}: {error_msg}",
                "model_used": model_name
            }

@mcp.tool(
    name="debug_chat",
    description="Debug version of chat to see raw Ollama response structure"
)
async def debug_chat(user_message: str = "Hi", model_name: str = None):
    """Debug chat to inspect the actual response structure from Ollama"""
    
    if not user_message:
        user_message = "Hi"  # Default simple message
    
    if not model_name:
        model_name = default_model
    
    if not await ensure_ollama_available():
        return {"status": "error", "message": "Ollama not available"}
    
    try:
        # Very short timeout for debugging
        response = await asyncio.wait_for(
            ollama_client.chat(
                model=model_name,
                messages=[{"role": "user", "content": user_message}],
                stream=False
            ),
            timeout=5.0  # Much shorter timeout
        )
        
        # Debug: Show what we actually get
        response_info = {
            "response_type": str(type(response)),
            "response_dir": [attr for attr in dir(response) if not attr.startswith('_')],
            "raw_response": str(response)[:500]  # First 500 chars
        }
        
        # Try to extract content
        if hasattr(response, 'message'):
            message_info = {
                "message_type": str(type(response.message)),
                "message_dir": [attr for attr in dir(response.message) if not attr.startswith('_')],
                "message_content": getattr(response.message, 'content', 'NO CONTENT ATTR')
            }
            response_info["message_info"] = message_info
        
        return {
            "status": "debug_success",
            "debug_info": response_info,
            "user_message": user_message,
            "model_used": model_name
        }
        
    except asyncio.TimeoutError:
        return {
            "status": "timeout_error",
            "message": f"❌ Ollama timeout after 5s with model {model_name}",
            "suggestion": "Model may be slow to load. Try: ollama run mistral (to preload)",
            "user_message": user_message,
            "model_used": model_name
        }
    except Exception as e:
        return {
            "status": "debug_error",
            "error": str(e),
            "error_type": str(type(e)),
            "user_message": user_message,
            "model_used": model_name
        }
    description="Stream chat responses in real-time (simulated streaming for MCP testing)"

async def stream_chat(user_message: str, model_name: str = None):
    """
    Simulated streaming function to test streaming capabilities
    In real implementation, this would yield chunks as they arrive
    """
    
    if not user_message:
        return {"status": "error", "message": "No message provided"}
    
    if not model_name:
        model_name = default_model
    
    if not await ensure_ollama_available():
        return {"status": "error", "message": "Ollama not available"}
    
    try:
        # Simulate streaming by breaking response into chunks
        response_data = await base_chat(user_message, model_name, stream=True)
        
        if response_data.get("status") == "success":
            full_response = response_data.get("response", "")
            
            # Simulate chunk streaming for demonstration
            words = full_response.split()
            chunks = []
            current_chunk = ""
            
            for i, word in enumerate(words):
                current_chunk += word + " "
                if len(current_chunk) > 50 or i == len(words) - 1:  # Chunk every ~50 chars
                    chunks.append(current_chunk.strip())
                    current_chunk = ""
            
            return {
                "status": "success",
                "message": "✅ Streaming simulation completed",
                "model_used": model_name,
                "total_chunks": len(chunks),
                "chunks": chunks,
                "full_response": full_response
            }
        else:
            return response_data
            
    except Exception as e:
        return {"status": "error", "message": f"Streaming error: {e}"}

@mcp.tool(
    name="simple_test",
    description="Extremely simple Ollama test without complex chat logic"
)
async def simple_test():
    """Simplest possible test to isolate the issue"""
    
    try:
        # Test 1: Check if client exists
        if not ollama_client:
            return "❌ No Ollama client"
        
        # Test 2: Try to list models (this worked before)
        models = await asyncio.wait_for(ollama_client.list(), timeout=3.0)
        model_count = len(models.models) if hasattr(models, 'models') else 0
        
        # Test 3: Try simplest possible chat call
        try:
            response = await asyncio.wait_for(
                ollama_client.chat(
                    model="mistral",  # Use explicit model name
                    messages=[{"role": "user", "content": "Say hi"}],
                    stream=False
                ),
                timeout=8.0
            )
            
            # Just check if we got anything back
            response_type = str(type(response))
            has_message = hasattr(response, 'message')
            
            return {
                "status": "success",
                "models_available": model_count,
                "chat_response_type": response_type,
                "has_message_attr": has_message,
                "message": "✅ Basic chat call succeeded!"
            }
            
        except asyncio.TimeoutError:
            return {
                "status": "chat_timeout",
                "models_available": model_count,
                "message": "❌ Chat timed out but model listing works"
            }
        except Exception as chat_error:
            return {
                "status": "chat_error", 
                "models_available": model_count,
                "chat_error": str(chat_error),
                "message": "❌ Chat failed but model listing works"
            }
        
    except Exception as e:
        return {
            "status": "connection_error",
            "error": str(e),
            "message": "❌ Basic Ollama connection failed"
        }
    description="Ultra-fast test of Ollama without actual chat"

async def quick_test():
    """Test Ollama connectivity without doing actual chat"""
    try:
        if not ollama_available:
            status = await check_ollama()
            if not status:
                return "❌ Ollama not available - service not running"
        
        # Just test if we can reach Ollama quickly
        models = await asyncio.wait_for(ollama_client.list(), timeout=3.0)
        model_count = len(models.models) if hasattr(models, 'models') else 0
        
        return f"✅ Ollama responsive - {model_count} models available"
        
    except asyncio.TimeoutError:
        return "❌ Ollama too slow (>3s) - may be overloaded"
    except Exception as e:
        return f"❌ Connection error: {e}"

@mcp.tool(
    name="check_model_status",
    description="Check if a model is loaded and ready for chat"
)
async def check_model_status(model_name: str = None):
    """Check if a specific model is loaded and responsive"""
    
    if not model_name:
        model_name = default_model
    
    if not await ensure_ollama_available():
        return {"status": "error", "message": "Ollama not available"}
    
    try:
        # Check if model exists in list
        models_result = await list_models()
        if models_result.get("status") != "success":
            return {"status": "error", "message": "Cannot check model list"}
        
        available_models = [m["name"] for m in models_result.get("models", [])]
        if model_name not in available_models:
            return {
                "status": "not_found",
                "message": f"❌ Model '{model_name}' not found",
                "available_models": available_models
            }
        
        # Try a very quick chat to see if it's loaded
        import time
        start_time = time.time()
        
        try:
            response = await asyncio.wait_for(
                ollama_client.chat(
                    model=model_name,
                    messages=[{"role": "user", "content": "1"}],  # Minimal input
                    stream=False
                ),
                timeout=5.0  # Very short timeout
            )
            
            response_time = time.time() - start_time
            
            return {
                "status": "loaded",
                "message": f"✅ Model '{model_name}' is loaded and ready",
                "model_name": model_name,
                "response_time_seconds": round(response_time, 2),
                "ready_for_chat": True
            }
            
        except asyncio.TimeoutError:
            return {
                "status": "loading",
                "message": f"⏳ Model '{model_name}' exists but is slow to respond (>5s)",
                "model_name": model_name,
                "suggestion": "Model may be loading into memory. Wait a moment and try again.",
                "ready_for_chat": False
            }
        
    except Exception as e:
        return {
            "status": "error",
            "message": f"❌ Error checking model status: {str(e)}",
            "model_name": model_name
        }
    description="Preload a model to avoid timeout issues during chat"

async def preload_model(model_name: str = None):
    """Preload a model by sending it a simple request to wake it up"""
    
    if not model_name:
        model_name = default_model
    
    if not await ensure_ollama_available():
        return {"status": "error", "message": "Ollama not available"}
    
    try:
        # Send a very simple message to preload the model
        print(f"🔄 Preloading model {model_name}...")
        
        response = await asyncio.wait_for(
            ollama_client.chat(
                model=model_name,
                messages=[{"role": "user", "content": "Hi"}],
                stream=False
            ),
            timeout=30.0  # Longer timeout for initial load
        )
        
        return {
            "status": "success",
            "message": f"✅ Model {model_name} preloaded successfully",
            "model_name": model_name,
            "response_received": True
        }
        
    except asyncio.TimeoutError:
        return {
            "status": "timeout",
            "message": f"⏱️ Model {model_name} took >30s to load - may be too large for system",
            "model_name": model_name,
            "suggestion": "Try a smaller model like 'mistral:7b' or check system resources"
        }
    except Exception as e:
        return {
            "status": "error",
            "message": f"❌ Failed to preload {model_name}: {str(e)}",
            "model_name": model_name,
            "error_type": str(type(e))
        }
    description="Set the default model for chat operations"

async def set_default_model(model_name: str):
    """Set the default model to use for chat operations"""
    global default_model
    
    # Verify the model exists
    models_result = await list_models()
    if models_result.get("status") == "success":
        available_models = [m["name"] for m in models_result.get("models", [])]
        if model_name in available_models:
            default_model = model_name
            return {
                "status": "success",
                "message": f"✅ Default model set to: {model_name}",
                "previous_default": default_model
            }
        else:
            return {
                "status": "error",
                "message": f"❌ Model '{model_name}' not found. Available: {', '.join(available_models)}"
            }
    else:
        return {
            "status": "error", 
            "message": "❌ Cannot verify model - Ollama connection issue"
        }

# Server startup with enhanced initialization
async def initialize_server():
    """Initialize server and check Ollama connectivity"""
    print("🚀 Starting MCP Server for Dual Chatbot Platform...")
    print("📊 Task 2 - Phase 1: Base LLM Integration with Streaming")
    print("🔧 Features: Enhanced base_chat tool, model selection, streaming responses")
    
    # Initialize Ollama connection
    print("🔍 Checking Ollama connectivity...")
    if await check_ollama():
        print("✅ Ollama connection successful!")
        
        # Try to list models to give user feedback
        models_result = await list_models()
        if models_result.get("status") == "success":
            print(f"📋 Found {models_result.get('count', 0)} available models")
        else:
            print("⚠️ Ollama connected but no models found - you may need to download models")
    else:
        print("❌ Ollama connection failed - server will still start but chat functions will be limited")
        print("💡 To fix: Ensure Ollama is running ('ollama serve' or start Ollama desktop app)")

if __name__ == "__main__":
    try:
        # Run initialization
        asyncio.run(initialize_server())
        
        # Run the MCP server
        mcp.run(transport='stdio')
        
    except KeyboardInterrupt:
        print("\n🛑 MCP Server shutting down...")
    except Exception as e:
        print(f"❌ Server error: {e}")
        logger.error(f"Server error: {e}")