# sqlalchemy-wrapper

A thin convenience wrapper around [SQLAlchemy 2.0](https://docs.sqlalchemy.org/en/20/) 
that handles engine and session lifecycle, multi-bind routing, and declarative
base setup — adapted from the patterns used in Flask-SQLAlchemy, for use outside
Flask.

## Usage

### Sync

```python
from sqlalchemy_wrapper import SQLAlchemyWrapper
from sqlalchemy.orm import DeclarativeBase

db = SQLAlchemyWrapper()
db.init("sqlite+aiosqlite:///app.db")

class User(db.Model):
    __tablename__ = "user"
    ...

db.create_all()

with db.scoped_session() as session:
    session.add(User(...))
    session.commit()
```

### Async

```python
from sqlalchemy_wrapper import AsyncSQLAlchemyWrapper

db = AsyncSQLAlchemyWrapper()
db.init("sqlite+aiosqlite:///app.db")

class User(db.Model):
    __tablename__ = "user"
    ...

async def main():
    await db.create_all()
    
    async with db.scoped_session() as session:
        session.add(User(...))
        await session.commit()
```

## Configuration

The `config` argument (or the argument to `db.init()`) can be:

| Type    | Expected content                                                      |
|---------|-----------------------------------------------------------------------|
| `str`   | Database URI directly                                                 |
| `dict`  | Must contain `SQLALCHEMY_DATABASE_URI`; optionally `SQLALCHEMY_BINDS` |
| `class` | Same keys as dict, as class attributes or callables                   |

`SQLALCHEMY_BINDS` is a `dict[str, str]` mapping bind names to URIs, used for
multi-database setups. Tag a model with `__bind_key__ = "name"` to route it to
the matching bind.

## API

Both `SQLAlchemyWrapper` and `AsyncSQLAlchemyWrapper` share the following
interface (async variants use `async with` / `await`):

| Member                | Description                                               |
|-----------------------|-----------------------------------------------------------|
| `db.Model`            | The declarative base for defining models                  |
| `db.session`          | The current scoped session                                |
| `db.engines`          | Dict of all engines keyed by bind name (`None` = default) |
| `db.init(config)`     | Finish initialization after construction                  |
| `db.scoped_session()` | Context manager — commits on exit, rolls back on error    |
| `db.create_all()`     | Create all tables                                         |
| `db.drop_all()`       | Drop all tables                                           |
| `db.close()`          | Remove the current scoped session                         |
| `db.rollback()`       | Roll back the current session                             |

## Credits

Adapted from the [Flask](https://flask.palletsprojects.com/en/stable/patterns/sqlalchemy/) 
and [Flask-SQLAlchemy](https://github.com/pallets-eco/flask-sqlalchemy/)
SQLAlchemy patterns.
