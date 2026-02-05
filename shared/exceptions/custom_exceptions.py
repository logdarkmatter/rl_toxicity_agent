class HTTPError(Exception):
    """Exception raised for HTTP errors."""

    def __init__(self, status_code: int, detail: str):
        self.status_code = status_code
        self.detail = detail
        super().__init__(f"HTTP {status_code}: {detail}")


class ClassifyMessageError(Exception):
    """ Exception raised for errors in the classify_message function."""

    def __init__(self, message: str):
        super().__init__(message)


class ModelLoadError(Exception):
    """Exception raised for errors in the model loading process."""

    def __init__(self, message: str):
        super().__init__(message)


class TextPreprocessingError(Exception):
    """ Exception raised for errors in the text preprocessing process."""

    def __init__(self, message: str):
        super().__init__(message)


class DataLoadingError(Exception):
    """Exception raised for errors in the data loading process."""

    def __init__(self, message: str):
        super().__init__(message)


class DataPreprocessingError(Exception):
    """Exception raised for errors in the data preprocessing process."""

    def __init__(self, message: str):
        super().__init__(message)


class EmbeddingGenerationError(Exception):
    """Exception raised for errors in the embedding generation process."""

    def __init__(self, message: str):
        super().__init__(message)


class ModelTrainingError(Exception):
    """Exception raised for errors in the model training process."""

    def __init__(self, message: str):
        super().__init__(message)


class ModelEvaluationError(Exception):
    """Exception raised for errors in the model evaluation process."""

    def __init__(self, message: str):
        super().__init__(message)


class ModelSavingError(Exception):
    """Exception raised for errors in the model saving process."""

    def __init__(self, message: str):
        super().__init__(message)


class CalculateAverageError(Exception):
    """Exception raised for errors in the calculating averages."""

    def __init__(self, message: str):
        super().__init__(message)


class DatabaseError(Exception):
    """Exception raised for errors in the database operations."""

    def __init__(self, message: str):
        super().__init__(message)
