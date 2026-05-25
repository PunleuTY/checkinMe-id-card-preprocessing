from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    app_name: str = "NID Preprocessor"
    debug: bool = False

    # Watermark
    watermark_text: str = "CheckinMe"
    watermark_opacity: float = 0.35

    # Output resolution for processed_image
    output_width: int = 1024
    output_height: int = 640

    # WebP quality (0-100)
    webp_quality: int = 85

    class Config:
        env_file = ".env"


settings = Settings()
