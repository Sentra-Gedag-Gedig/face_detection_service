from pydantic import BaseModel
from typing import Dict, List, Optional


class Position(BaseModel):
    x: int
    y: int


class DetectionResponse(BaseModel):
    status: str
    instructions: List[str]
    face_position: Optional[Position] = None
    face_size: Optional[float] = None
    frame_center: Position
    deviations: Optional[Dict[str, float]] = None
