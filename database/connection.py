"""Database connection management module."""
import pymysql
from typing import Optional, Dict, Any
from contextlib import contextmanager
from utils.logger import setup_logger, log_with_context
from config.settings import settings


logger = setup_logger(__name__)


class DatabaseConnection:
    """Manage MySQL database connections."""

    def __init__(self):
        """Initialize database connection manager."""
        self.config = settings.get_db_config()
        self._connection: Optional[pymysql.Connection] = None

    def connect(self) -> pymysql.Connection:
        """Create database connection."""
        try:
            self._connection = pymysql.connect(**self.config)
            log_with_context(
                logger, 'info',
                "Database connection established",
                host=self.config['host'],
                database=self.config['database']
            )
            return self._connection
        except pymysql.Error as e:
            log_with_context(
                logger, 'error',
                "Failed to connect to database",
                error=str(e),
                error_code=e.args[0] if e.args else None
            )
            raise

    def close(self):
        """Close database connection."""
        if self._connection:
            self._connection.close()
            self._connection = None
            logger.info("Database connection closed")

    @contextmanager
    def get_cursor(self, cursor_class=pymysql.cursors.DictCursor):
        """Get database cursor context manager."""
        connection = self.connect()
        try:
            with connection.cursor(cursor_class) as cursor:
                yield cursor
                connection.commit()
        except Exception as e:
            connection.rollback()
            log_with_context(
                logger, 'error',
                "Database operation failed, rolling back",
                error=str(e)
            )
            raise
        finally:
            self.close()


# Global connection instance
db = DatabaseConnection()


@contextmanager
def get_db_cursor(cursor_class=pymysql.cursors.DictCursor):
    """Convenience function to get database cursor."""
    with db.get_cursor(cursor_class) as cursor:
        yield cursor