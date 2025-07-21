import json
from fastmcp import Client
from langchain_mcp_adapters.client import MultiServerMCPClient
import asyncio
from typing import Optional
from contextlib import AsyncExitStack
import os
import logging
import httpx
from typing import Optional
from langgraph.prebuilt import create_react_agent
from mcp.client.stdio import stdio_client
from mcp import ClientSession, StdioServerParameters

# Set up logging
logging.basicConfig(level=os.getenv('LOG_LEVEL', 'INFO'))
logger = logging.getLogger(__name__)

from langchain_ollama.chat_models import ChatOllama
llm = ChatOllama(model="llama3.1")
mcp_client = Client("http://localhost:8000/sse")

# Custom JSON encoder for objects with 'content' attribute
class CustomEncoder(json.JSONEncoder):
    def default(self, o):
        if hasattr(o, "content"):
            return {"type": o.__class__.__name__, "content": o.content}
        return super().default(o)

async def check_server_health(url: str) -> bool:
    """Check if the MCP server is running and responding correctly."""
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            async with client.stream("GET", url) as response:
                content_type = response.headers.get('content-type', '')
                if 'text/event-stream' in content_type and response.status_code == 200:
                    logger.info(f"✅ MCP server is running correctly at {url}")
                    return True
                elif response.status_code == 200:
                    logger.warning(f"⚠️ Server responded with status 200 but wrong content type: {content_type}")
                    return False
                else:
                    logger.error(f"❌ Server responded with status {response.status_code}")
                    return False
    except httpx.ConnectError:
        logger.error(f"❌ Cannot connect to MCP server at {url}")
        return False
    except Exception as e:
        logger.error(f"❌ Error checking server health: {e}")
        return False


async def main():
    # Get the MCP server URL from environment or use default
    mcp_url = os.getenv("REMOTE_MCP_URL", "http://localhost:8000/sse")

    logger.info(f"🔍 Checking MCP server at: {mcp_url}")

    # Check if server is healthy before proceeding
    if not await check_server_health(mcp_url):
        logger.error("❌ MCP server is not available. Please ensure:")
        logger.error("   1. The MCP server is running")
        logger.error("   2. The URL is correct")
        logger.error("   3. The server supports SSE transport")
        logger.error(f"   Current URL: {mcp_url}")
        return

    try:
        client = MultiServerMCPClient({
            "mcp_server": {
                "url": mcp_url,
                "transport": "sse"
            }
        })

        logger.info("🔄 Connecting to MCP server...")
        tools = await client.get_tools()
        logger.info("Received Tools..")
        print(tools)
        if tools:
            logger.info(f"✅ Successfully loaded {len(tools)} MCP tools:")
            for tool in tools:
                print(f"  - {tool.name}")
        else:
            logger.warning("⚠️ No tools found from MCP server")

        agent = create_react_agent(llm, tools)
        print("MCP Client Started! Type 'quit' to exit.")
        response = await agent.ainvoke({"messages": "what is weather in Dallas, TX"})
        try:
            formatted = json.dumps(response, indent=2, cls=CustomEncoder)
        except Exception:
            formatted = str(response)
        print("\\nResponse:")
        print(formatted)

        # await test_client()

    except Exception as e:
        logger.error(f"❌ Error connecting to MCP server: {e}")
        logger.error("Please check:")
        logger.error("   1. Server is running and accessible")
        logger.error("   2. Network connectivity")
        logger.error("   3. Server configuration")

async def test_client():
    async with mcp_client:
        # Basic server interaction
        await mcp_client.ping()

        # List available operations
        tools = await mcp_client.list_tools()
        print(tools)

        result = await mcp_client.call_tool("get_weather", {"location": "Dallas, TX"})
        print(result)

if __name__ == "__main__":
    asyncio.run(main())
    # asyncio.run(test_client())
