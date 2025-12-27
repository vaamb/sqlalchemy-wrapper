from __future__ import annotations

from asyncio import current_task
from contextlib import asynccontextmanager, contextmanager
from dataclasses import dataclass
from typing import Any, AsyncGenerator, Generator, Type, TypedDict
import warnings

from sqlalchemy.engine import create_engine, Engine
from sqlalchemy.ext.asyncio import (
    async_scoped_session, async_sessionmaker, AsyncEngine, AsyncSession,
    create_async_engine)
from sqlalchemy.orm import (
    declarative_base, DeclarativeBase, DeclarativeMeta, scoped_session,
    Session, sessionmaker)
from sqlalchemy.sql.schema import MetaData

from .base import CustomMeta


class ConfigDict(TypedDict):
    SQLALCHEMY_DATABASE_URI: str
    SQLALCHEMY_BINDS: dict[str, str]


def _config_dict_from_class(obj: Type) -> ConfigDict:
    cfg = {}
    for key in dir(obj):
        if key not in ("SQLALCHEMY_DATABASE_URI", "SQLALCHEMY_BINDS"):
            continue
        attr = getattr(obj, key)
        if callable(attr):
            cfg[key] = attr()
        else:
            cfg[key] = attr
    return cfg


@dataclass
class Config:
    uri: str
    binds: dict[str, str]


class SQLAlchemyWrapper:
    """Convenience wrapper to use SQLAlchemy

    To use it, you need to provide a config object either during initialisation
    or after using `sql_alchemy_wrapper.init(config_object). This config object
    can either be a class, a dict or a str.
    If providing a class or a dict config, it must contain the field
    `SQLALCHEMY_DATABASE_URI` with the database uri. It can also provide the
    field `SQLALCHEMY_DATABASE_URI` which contains a dict with the name of
    the bindings and the binding uri.
    If using a str, it must be the database uri.

    For a safe use, use as follows:
    ``
    db = SQLAlchemyWrapper()
    db.init("sqlite:///")
    with db.scoped_session() as session:
        stmt = your_statement_here
        result = session.execute(stmt)
    ``
    This will automatically create a scoped session and remove it at the end of
    the scope.
    """

    def __init__(
            self,
            config: type | str | dict | None = None,
            model: Type[DeclarativeBase] | Type[DeclarativeMeta] | None = None,
            metadata: Metadata | None = None,
            engine_options: dict | None = None,
            session_options: dict | None = None,
    ):
        """Construct a new `SQLAlchemyWrapper` instance

        :param config: a str, a class or a dict with the database(s)
        address(es). Cf the `init` method for a better description of
        """
        self.Model: DeclarativeBase | DeclarativeMeta = \
            self._create_declarative_base(model, metadata)
        self._engine_options = engine_options or {}
        self._session_options = session_options or {}
        self._session_factory: sessionmaker | None = None
        self._session: scoped_session | None = None
        self._engines: dict[str | None, Engine] = {}
        self._config: Config | None = None

        if config:
            self.init(config)

    @property
    def session(self) -> Session:
        """An SQLAlchemy session to manage ORM-objects"""
        if self._session is None:
            raise RuntimeError(
                "No config option was provided. Use db.init(config) to finish "
                "db initialization"
            )
        return self._session()

    @property
    def engines(self) -> dict[str | None, Engine]:
        if self._engines is None:
            raise RuntimeError(
                "No config option was provided. Use db.init(config) to finish "
                "db initialization"
            )
        return self._engines

    @property
    def config(self) -> Config:
        """The config with the main uri and optional secondary bindings uris"""
        if self._config is None:
            raise RuntimeError(
                "No config option was provided. Use db.init(config) to finish "
                "db initialization"
            )
        return self._config

    def init(self, config_object: type | str | dict) -> None:
        """Finish the configuration and the initialization of the wrapper to
        use it. This must be called before accessing the database session.

        :param config_object: a str, a class or a dict with the database(s)
        address(es). If a str is provided, it must be be a valid database
        address. If a class or a dict is provided, it must contain the field
        `SQLALCHEMY_DATABASE_URI` with the database uri. It can also provide
        the field `SQLALCHEMY_DATABASE_URI` which contains a dict with the name
        of the bindings as keys and the binding uri as values.
        """
        self._init_config(config_object)
        self._create_session_factory()

    def _create_declarative_base(
            self,
            model: Type[DeclarativeBase] | Type[DeclarativeMeta] | None,
            metadata: Metadata | None,
    ) -> Any:
        if model is None:
            return declarative_base(metaclass=CustomMeta, metadata=metadata)
        elif issubclass(model, DeclarativeMeta):
            return declarative_base(metaclass=model, metadata=metadata)
        elif issubclass(model, DeclarativeBase):
            if metadata is not None:
                model.metadata = metadata
            return model
        else:
            raise TypeError("model is not a valid SQLAlchemy declarative base")

    def _init_config(self, config_object: type | str | dict) -> None:
        if isinstance(config_object, type):
            config = _config_dict_from_class(config_object)
        elif isinstance(config_object, str):
            config = {"SQLALCHEMY_DATABASE_URI": config_object}
        elif isinstance(config_object, dict):
            config = config_object
        else:
            raise TypeError(
                "config_object can either be a str, a dict or a class"
            )
        if "SQLALCHEMY_DATABASE_URI" not in config:
            raise RuntimeError(
                "Config object needs the parameter 'SQLALCHEMY_DATABASE_URI'"
            )
        self._config = Config(
            uri=config["SQLALCHEMY_DATABASE_URI"],
            binds=config.get("SQLALCHEMY_BINDS", {})
        )

    def _create_session_factory(self) -> None:
        self._session_factory = sessionmaker(
            binds=self.get_binds_mapping(),
            **self._session_options,
        )
        self._session = scoped_session(self._session_factory)

    def _create_engine(self, uri, **kwargs) -> Engine:
        return create_engine(uri, **self._engine_options, **kwargs)

    def get_binds_list(self) -> list[str | None]:
        return [None, *list(self.config.binds.keys())]

    def get_tables_for_bind(self, bind: str = None) -> list:
        return [
            table for table in self.Model.metadata.tables.values()
            if table.info.get("bind_key", None) == bind
        ]

    def get_uri_for_bind(self, bind: str = None) -> str:
        if bind is None:
            return self.config.uri
        binds: dict = self.config.binds
        if bind not in binds:
            warnings.warn(
                f"Bind {bind} is not defined by 'SQLALCHEMY_BINDS' in the "
                f"config, using 'SQLALCHEMY_DATABASE_URI' instead"
            )
            return self.config.uri
        return binds[bind]

    def get_engine_for_bind(self, bind: str = None) -> Engine:
        engine = self._engines.get(bind, None)
        if engine is None:
            engine = self._create_engine(
                self.get_uri_for_bind(bind),
            )
            self._engines[bind] = engine
        return engine

    @contextmanager
    def scoped_session(self) -> Generator[Session, None, None]:
        """Provide a scoped session context that automatically tries to commit
        at the end of its scope.
        """
        session = self.session
        try:
            yield session
            session.commit()
        except Exception as e:
            self.rollback()
            raise e
        finally:
            self.close()

    def get_binds_mapping(self) -> dict:
        """Provides a dict with all the linked tables as keys and their engine
        as values
        """
        binds = self.get_binds_list()
        result = {}
        for bind in binds:
            engine = self.get_engine_for_bind(bind)
            result.update(
                {table: engine for table in self.get_tables_for_bind(bind)})
        return result

    def create_all(self) -> None:
        """Create all the tables linked to `self.Model`
        """
        binds = self.get_binds_list()
        for bind in binds:
            engine = self.get_engine_for_bind(bind)
            tables = self.get_tables_for_bind(bind)
            self.Model.metadata.create_all(bind=engine, tables=tables)

    def drop_all(self) -> None:
        """Drop all the tables linked to `self.Model`
        """
        binds = self.get_binds_list()
        for bind in binds:
            engine = self.get_engine_for_bind(bind)
            tables = self.get_tables_for_bind(bind)
            self.Model.metadata.drop_all(bind=engine, tables=tables)

    def close(self) -> None:
        """Close the current session
        """
        if self._session is None:
            raise RuntimeError(
                "No config option was provided. Use db.init(config) to finish "
                "db initialization"
            )
        self._session.remove()

    def rollback(self) -> None:
        """Rollback the current session
        """
        if self._session is None:
            raise RuntimeError(
                "No config option was provided. Use db.init(config) to finish "
                "db initialization"
            )
        self._session.rollback()


class AsyncSQLAlchemyWrapper(SQLAlchemyWrapper):
    """Convenience wrapper to use SQLAlchemy

    To use it, you need to provide a config object either during initialisation
    or after using `sql_alchemy_wrapper.init(config_object). This config object
    can either be a class, a dict or a str.
    If providing a class or a dict config, it must contain the field
    `SQLALCHEMY_DATABASE_URI` with the database uri. It can also provide the
    field `SQLALCHEMY_DATABASE_URI` which contains a dict with the name of
    the bindings and the binding uri.
    If using a str, it must be the database uri.

    For a safe use, use as follows:
    ``
    db = SQLAlchemyWrapper()
    db.init("sqlite:///")
    with db.scoped_session() as session:
        stmt = your_statement_here
        result = session.execute(stmt)
    ``
    This will automatically create a scoped session and remove it at the end of
    the scope.
    """
    _session_factory: async_sessionmaker | None
    _session: async_scoped_session | None
    _engines: dict[str | None, AsyncEngine]

    def _create_session_factory(self) -> None:
        self._session_factory = async_sessionmaker(
            binds=self.get_binds_mapping(),
            **self._session_options,
        )
        self._session = async_scoped_session(
            self._session_factory, current_task)

    def _create_engine(self, uri, **kwargs) -> AsyncEngine:
        return create_async_engine(uri, **self._engine_options, **kwargs)

    @property
    def session(self) -> AsyncSession:
        """An SQLAlchemy async session to manage ORM-objects"""
        if not self._session:
            raise RuntimeError(
                "No config option was provided. Use db.init(config) to finish "
                "db initialization"
            )
        else:
            return self._session()

    @property
    def engines(self) -> dict[str | None, AsyncEngine]:
        if self._engines is None:
            raise RuntimeError(
                "No config option was provided. Use db.init(config) to finish "
                "db initialization"
            )
        return self._engines

    @asynccontextmanager
    async def scoped_session(self) -> AsyncGenerator[AsyncSession, None, None]:
        """Provide an async scoped session context that automatically tries to
        commit at the end of its scope.
        """
        session = self.session
        try:
            yield session
            await session.commit()
        except Exception as e:
            await self.rollback()
            raise e
        finally:
            await self.close()

    async def create_all(self) -> None:
        """Create all the tables linked to `self.Model`
        """
        binds = self.get_binds_list()
        for bind in binds:
            engine = self.get_engine_for_bind(bind)
            tables = self.get_tables_for_bind(bind)
            async with engine.begin() as conn:
                await conn.run_sync(
                    self.Model.metadata.create_all, tables=tables
                )

    async def drop_all(self) -> None:
        """Drop all the tables linked to `self.Model`
        """
        binds = self.get_binds_list()
        for bind in binds:
            engine = self.get_engine_for_bind(bind)
            tables = self.get_tables_for_bind(bind)
            async with engine.begin() as conn:
                await conn.run_sync(
                    self.Model.metadata.drop_all, tables=tables
                )

    async def close(self) -> None:
        """Close the current session
        """
        if self._session is None:
            raise RuntimeError(
                "No config option was provided. Use db.init(config) to finish "
                "db initialization"
            )
        await self._session.remove()

    async def rollback(self) -> None:
        """Rollback the current session
        """
        if self._session is None:
            raise RuntimeError(
                "No config option was provided. Use db.init(config) to finish "
                "db initialization"
            )
        await self._session.rollback()
