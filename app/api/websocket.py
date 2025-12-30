"""
Voice WebSocket Handler
"""
import sys
import os
import asyncio
import json
import base64
import time
from uuid import UUID
from fastapi import WebSocket, WebSocketDisconnect, APIRouter
from sqlalchemy import select

# Add src to path for voice_agent imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'src'))

from voice_agent.providers.stt.deepgram_streaming import create_streaming_stt
from voice_agent.providers.llm.openai_realtime import create_openai_realtime
from voice_agent.providers.tts.cartesia_streaming import create_streaming_tts

from app.core.database import async_session_maker
from app.models.agent import Agent

# WebSocket Router
ws_router = APIRouter()


@ws_router.websocket("/ws/{agent_id}")
async def voice_websocket(websocket: WebSocket, agent_id: str):
    """Handle voice WebSocket connection for an agent."""
    await websocket.accept()

    # Get agent config using SQLAlchemy async session
    async with async_session_maker() as session:
        result = await session.execute(
            select(Agent).where(Agent.id == agent_id)
        )
        agent_model = result.scalar_one_or_none()

    if not agent_model:
        await websocket.send_json({"type": "error", "message": "Agent not found"})
        await websocket.close()
        return

    # Convert to dict for easier access
    agent = {
        "id": str(agent_model.id),
        "name": agent_model.name,
        "system_prompt": agent_model.system_prompt,
        "language": agent_model.language,
        "voice_id": agent_model.voice_id,
        "voice_speed": agent_model.voice_speed,
        "stt_provider": agent_model.stt_provider,
        "llm_provider": agent_model.llm_provider,
        "llm_model": agent_model.llm_model,
        "tts_provider": agent_model.tts_provider,
    }
    print(f"[WS] Agent loaded: {agent['name']}")

    stt = None
    llm = None
    tts = None
    processing = False

    async def send(data):
        try:
            await websocket.send_json(data)
        except:
            pass

    async def on_transcript(result):
        nonlocal processing
        print(f"[STT] '{result.text}' (final={result.is_final})")
        await send({"type": "transcript", "text": result.text, "is_final": result.is_final})

        if result.is_final and result.text.strip() and not processing:
            processing = True
            try:
                # LLM
                print(f"[LLM] Sending: '{result.text}'")
                t0 = time.time()
                response, llm_ms = await llm.send_text(result.text)
                print(f"[LLM] Response: '{response}' ({llm_ms:.0f}ms)")

                # TTS
                print(f"[TTS] Starting...")
                t1 = time.time()
                chunks = []
                tts_ttfb = None
                async for chunk in tts.synthesize_stream(response):
                    if tts_ttfb is None:
                        tts_ttfb = (time.time() - t1) * 1000
                        print(f"[TTS] First chunk ({tts_ttfb:.0f}ms)")
                    chunks.append(chunk)

                print(f"[TTS] Total: {len(chunks)} chunks, {sum(len(c) for c in chunks)} bytes")

                ttfa = llm_ms + (tts_ttfb or 0)

                await send({
                    "type": "response",
                    "text": response,
                    "metrics": {"ttfa": ttfa, "llm": llm_ms, "tts": tts_ttfb or 0}
                })

                if chunks:
                    audio_data = b"".join(chunks)
                    print(f"[AUDIO] Sending {len(audio_data)} bytes")
                    await send({
                        "type": "audio",
                        "audio": base64.b64encode(audio_data).decode()
                    })

            except Exception as e:
                import traceback
                print(f"[ERROR] {e}")
                traceback.print_exc()
                await send({"type": "error", "message": str(e)})
            finally:
                processing = False

    try:
        # Connect STT
        await send({"type": "status", "component": "stt", "status": "connecting"})
        stt = create_streaming_stt(model="nova-2", language=agent['language'])
        await stt.connect(on_transcript=on_transcript)
        await send({"type": "status", "component": "stt", "status": "connected"})

        # Connect LLM
        await send({"type": "status", "component": "llm", "status": "connecting"})
        llm = create_openai_realtime(
            model=agent['llm_model'],
            instructions=agent['system_prompt']
        )
        await llm.connect()
        await send({"type": "status", "component": "llm", "status": "connected"})

        # Connect TTS
        await send({"type": "status", "component": "tts", "status": "connecting"})
        tts = create_streaming_tts(
            voice=agent['voice_id'],
            speed=agent['voice_speed']
        )
        await tts.connect()
        await send({"type": "status", "component": "tts", "status": "connected"})

        await send({"type": "status", "component": "ready", "status": "ready"})
        await send({"type": "agent", "name": agent['name']})

        # Receive audio
        while True:
            data = await websocket.receive_bytes()
            if stt:
                await stt.send_audio(data)

    except WebSocketDisconnect:
        print("[WS] Client disconnected")
    except Exception as e:
        print(f"[WS] Error: {e}")
        await send({"type": "error", "message": str(e)})
    finally:
        if stt:
            await stt.close()
        if llm:
            await llm.close()
        if tts:
            await tts.close()
