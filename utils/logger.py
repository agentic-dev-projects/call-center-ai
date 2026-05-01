from loguru import logger

logger.add(
    "logs/app.log",
    rotation="1 MB",
    level="INFO",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level:<8} | {name}:{line} | {message}"
)
