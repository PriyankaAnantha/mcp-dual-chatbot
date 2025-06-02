from mcp.server.fastmcp import FastMCP

# 1. Initialize FastMCP server with a standard name
mcp = FastMCP(name="DualChatbotPlatform_MCP")

# No tools or resources defined yet for Task 1.

# 2. Main execution block to run the server
if __name__ == "__main__":
    print("Starting MCP Server for Dual Chatbot Platform...")
    try:
        # For stdio transport
        mcp.run(transport='stdio')
    except KeyboardInterrupt:
        print("\nMCP Server shutting down...")
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        print("MCP Server stopped.")