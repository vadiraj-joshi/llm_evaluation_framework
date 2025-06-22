from typing import Optional, Dict
from ports.authentication_service import IAuthenticationService

class BasicAuthAdapter(IAuthenticationService):
    """
    A basic in-memory authentication adapter for demonstration purposes.
    In a real application, this would interact with a user database or
    an identity provider.
    """
    def __init__(self, users: Dict[str, str]):
        """
        Initializes the BasicAuthAdapter with a dictionary of valid users.
        :param users: A dictionary where keys are usernames and values are passwords.
                      (In production, passwords should be hashed).
        """
        self.users = users
        # For simplicity, a dummy token is returned for any successful authentication.
        # In a real system, this would be a JWT or session token.
        self.dummy_token_map: Dict[str, str] = {} # username -> dummy_token

    def authenticate_user(self, username: str, password: str) -> Optional[str]:
        """
        Authenticates a user using basic username and password check.
        :param username: The user's username.
        :param password: The user's password.
        :return: A dummy token (representing a successful login) if credentials are valid, None otherwise.
        """
        if self.users.get(username) == password:
            # Simulate token generation. In a real system, generate a secure token.
            token = f"dummy_token_for_{username}"
            self.dummy_token_map[username] = token
            return token
        return None

    def get_current_user_id(self, token: str) -> Optional[str]:
        """
        Retrieves the user ID associated with a given dummy token.
        :param token: The dummy authentication token.
        :return: The username (as user ID) if the token is valid, None otherwise.
        """
        for username, stored_token in self.dummy_token_map.items():
            if stored_token == token:
                return username
        return None
