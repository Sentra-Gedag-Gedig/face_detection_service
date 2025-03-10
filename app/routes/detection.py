from fastapi import WebSocket, APIRouter
from services.face_detector import FaceDetectionService
import numpy as np
import cv2

router = APIRouter()
detector = FaceDetectionService()


@router.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_bytes()

            # Decode
            frame = cv2.imdecode(np.frombuffer(data, np.uint8),
                                 cv2.IMREAD_COLOR)

            result = detector.process(frame)

            await websocket.send_json(result.dict())
    except Exception as e:
        print(f"WebSocket error: {e}")
    finally:
        await websocket.close()
