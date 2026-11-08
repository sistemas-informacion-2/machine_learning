from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Ajustes(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    ml_database_url: str = ""
    ml_clave_interna: str = ""


@lru_cache
def ajustes() -> Ajustes:
    return Ajustes()
