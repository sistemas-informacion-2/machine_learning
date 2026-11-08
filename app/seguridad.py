import hmac

from fastapi import Header, HTTPException, status

from app.config import ajustes


def verificar_clave(x_clave_interna: str = Header(default="")) -> None:
    """Solo NestJS conoce la clave. Sin clave configurada se rechaza todo: el servicio nunca queda abierto."""
    esperada = ajustes().ml_clave_interna
    if not esperada or not hmac.compare_digest(x_clave_interna.encode(), esperada.encode()):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Clave interna invalida")
