"""Domain-specific errors used by core ML components."""


class AudioProcessingError(Exception):
    """Structured error raised for invalid or unreadable audio inputs."""

    def __init__(self, error_code: str, description: str) -> None:
        if not error_code:
            raise ValueError("error_code must be a non-empty string")
        if not description:
            raise ValueError("description must be a non-empty string")

        self.error_code = error_code
        self.description = description
        super().__init__(f"{error_code}: {description}")