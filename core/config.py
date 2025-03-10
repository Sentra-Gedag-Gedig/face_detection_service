from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    center_deviation: float = 0.15
    min_face_ratio: float = 0.25
    max_face_ratio: float = 0.35
    cascade_path: str = "haarcascade_frontalface_default.xml"

    class Config:
        env_file = ".env"


settings = Settings()
