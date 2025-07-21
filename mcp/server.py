from mcp.server.fastmcp import FastMCP
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import json
import asyncio
from typing import AsyncGenerator
import uvicorn

# Create an MCP server
mcp = FastMCP("Weather Service")

# Create FastAPI app
app = FastAPI(title="MCP Server", description="Model Context Protocol Server with SSE support")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Tool implementation
@mcp.tool()
def get_weather(location: str) -> str:
    """Get the current weather for a specified location."""
    return f"Weather in {location}: Sunny, 80°F"


# Resource implementation
@mcp.resource("resource://{location}")
def weather_resource(location: str) -> str:
    """Provide weather data as a resource."""
    return f"Weather data for {location}: Sunny, 72°F"


# Prompt implementation
@mcp.prompt()
def weather_report(location: str) -> str:
    """Create a weather report prompt."""
    return f"""You are a weather reporter. Weather report for {location}?"""

# SSE endpoint for MCP clients
@app.get("/sse")
async def sse_endpoint():
    """Server-Sent Events endpoint for MCP clients."""

    async def event_stream() -> AsyncGenerator[str, None]:
        """Generate SSE events."""
        # Send initial connection event
        yield json.dumps("{\"type\": \"connection\", \"message\": \"MCP Server SSE stream connected\"}\n\n")

        # Send available tools
        tools_data = {
            "type": "tools",
            "tools": [
                {"name": "get_weather", "description": "Get the current weather for a specified location"},
                {"name": "weather_resource", "description": "Provide weather data as a resource"},
                {"name": "weather_report", "description": "Create a weather report prompt"}
            ]
        }
        yield f"{json.dumps(tools_data)}\n\n"

        # Keep connection alive with periodic heartbeat
        while True:
            await asyncio.sleep(30)  # Send heartbeat every 30 seconds
            heartbeat = {
                "type": "heartbeat",
                "timestamp": asyncio.get_event_loop().time()
            }
            yield f"{json.dumps(heartbeat)}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Headers": "Cache-Control"
        }
    )


# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "MCP Weather Service",
        "endpoints": {
            "html": "/",
            "sse": "/sse",
            "docs": "/docs"
        }
    }


# Run MCP server separately - it will handle its own endpoints
# The FastAPI app will handle HTML and SSE endpoints


# Run the server
if __name__ == "__main__":
    print("🚀 Starting MCP Server...")
    print("📱 HTML Interface: http://localhost:8000/")
    print("📡 SSE Endpoint: http://localhost:8000/sse")
    print("📚 API Docs: http://localhost:8000/docs")
    print("🏥 Health Check: http://localhost:8000/health")

    mcp.run(transport="sse")
