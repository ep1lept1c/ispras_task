from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import SecretStr
import pathlib
class Settings(BaseSettings):
    bot_token: SecretStr
    model_config = SettingsConfigDict(env_file= f"{pathlib.Path(__file__).resolve().parent}/.env", env_file_encoding='utf-8')
config = Settings()