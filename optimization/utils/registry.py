# utils/registry.py
from typing import Callable, TypeVar

T = TypeVar("T")

class Registry(dict):
    """既是 dict，又支持 .register(name) 装饰器的简易注册表。"""
    def register(self, name: str) -> Callable[[T], T]:
        def deco(obj: T) -> T:
            self[name] = obj
            return obj
        return deco

# —— 全局注册表：模型 & 元模块（注意：这里是 Registry 实例，不是 dict）
MODEL_REGISTRY: Registry = Registry()
META_REGISTRY:  Registry = Registry()

# —— 可选：向后兼容的装饰器别名（@register_model 等价于 @MODEL_REGISTRY.register）
def register_model(name: str) -> Callable[[T], T]:
    return MODEL_REGISTRY.register(name)

def register_meta(name: str) -> Callable[[T], T]:
    return META_REGISTRY.register(name)
