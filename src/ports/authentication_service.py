from abc import ABC, abstractmethod
from typing import Optional

class IAuthenticationService(ABC):
    """
    Abstract base class defining the port for an authentication service.
    This service is responsible for validating user credentials.
    """
    @abstractmethod
    def authenticate_user(self, username: str, password: str) -> Optional[str]:
        """
        Authenticates a user based on provided credentials.
        :param username: The user's username.
        :param password: The user's password.
        :return: A user ID or token if authentication is successful, None otherwise.
        """
        pass

    @abstractmethod
    def get_current_user_id(self, token: str) -> Optional[str]:
        """
        Retrieves the user ID associated with a given authentication token/credential.
        :param token: The authentication token or credential.
        :return: The user ID if valid, None otherwise.
        """
        pass
