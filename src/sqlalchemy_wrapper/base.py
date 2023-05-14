from __future__ import annotations

from typing import Any

from sqlalchemy.orm import DeclarativeMeta


class BindMetaMixin(type):
    # From Flask-SQLAlchemy
    def __init__(cls, classname: Any, bases: Any, dict_: Any) -> None:
        bind_key = (
            dict_.pop('__bind_key__', None)
            or getattr(cls, '__bind_key__', None)
        )

        super(BindMetaMixin, cls).__init__(classname, bases, dict_)

        if bind_key is not None and getattr(cls, '__table__', None) is not None:
            cls.__table__.info['bind_key'] = bind_key


class CustomMeta(BindMetaMixin, DeclarativeMeta):
    pass
