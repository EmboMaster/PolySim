# rpc_api.py
from typing import Dict, Any

_SERVICE = None  

def register_service(service) -> None:
    global _SERVICE
    _SERVICE = service

def step(d: Dict[str, Any]):
    s = _SERVICE
    if s is None:
        raise RuntimeError("Service not registered on server.")
    return s.step(d)


def reset_all():
    s = _SERVICE
    if s is None:
        raise RuntimeError("Service not registered on server.")
    return s.reset_all()

def set_is_evaluating():
    s = _SERVICE
    if s is None:
        raise RuntimeError("Service not registered on server.")
    return s.set_is_evaluating()

def set_episode_length_buf(episode_length_buf):
    s = _SERVICE
    if s is None:
        raise RuntimeError("Service not registered on server.")
    return s.set_episode_length_buf(episode_length_buf)

def get_episode_length_buf():
    s = _SERVICE
    if s is None:
        raise RuntimeError("Service not registered on server.")
    return s.get_episode_length_buf()
