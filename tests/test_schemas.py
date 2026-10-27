import pytest
from pydantic import ValidationError

from llm_client import ChatMessage, ModelConfig


@pytest.mark.parametrize("temperature", [0, 1, 2])
def test_temperatura_valida(temperature):
    assert ModelConfig(temperature=temperature).temperature == temperature


@pytest.mark.parametrize("temperature", [-0.1, 2.1])
def test_temperatura_fuera_de_rango(temperature):
    with pytest.raises(ValidationError):
        ModelConfig(temperature=temperature)


@pytest.mark.parametrize("max_tokens", [0, -5])
def test_max_tokens_debe_ser_positivo(max_tokens):
    with pytest.raises(ValidationError):
        ModelConfig(max_tokens=max_tokens)


def test_rol_invalido():
    with pytest.raises(ValidationError):
        ChatMessage(role="robot", content="hola")


def test_contenido_vacio():
    with pytest.raises(ValidationError):
        ChatMessage(role="user", content="")
