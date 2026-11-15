# Alembic's helpers for logging and running migrations
from logging.config import fileConfig

from alembic import context
from sqlalchemy import engine_from_config, pool

# Use the same database URL as the app (read from .env), so there is only one copy to maintain
from app.core.config import settings
from app.db.base import Base
# Importing the models registers every table with Base.metadata
import app.models  # noqa: F401

config = context.config
# "%" is escaped because Alembic's config treats it specially
config.set_main_option("sqlalchemy.url", settings.database_url.replace("%", "%%"))
if config.config_file_name:
    fileConfig(config.config_file_name)

# The tables Alembic compares the real database against
target_metadata = Base.metadata


def run_migrations_offline() -> None:
    # Generates SQL text without connecting to the database
    context.configure(
        url=config.get_main_option("sqlalchemy.url"),
        target_metadata=target_metadata,
        literal_binds=True,
        compare_type=True,
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    # Connects to PostgreSQL and applies the migrations (the normal mode)
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )
    with connectable.connect() as connection:
        # compare_type=True also detects column type changes in future migrations
        context.configure(connection=connection, target_metadata=target_metadata, compare_type=True)
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()