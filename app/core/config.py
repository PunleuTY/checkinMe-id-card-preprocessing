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

    # Gemini (experiments)
    gemini_api_key: str = ""
    gemini_model: str = "gemini-2.5-flash"

    class Config:
        env_file = ".env"
        extra = "ignore"  # tolerate unrelated keys in a shared .env (e.g. Laravel GEMINI_SERVICE_*)


settings = Settings()
