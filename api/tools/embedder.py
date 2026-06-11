"""Embedder factory — returns a configured embedder instance.

No longer depends on adalflow.
"""

from api.config import configs, get_embedder_type


class Embedder:
    """Minimal embedder wrapping a model client for OpenAI-compatible embedding calls.

    Replaces adalflow.Embedder.
    """

    def __init__(self, model_client, model_kwargs):
        self.model_client = model_client
        self.model_kwargs = model_kwargs
        self.batch_size = 500

    def __call__(self, input):
        """Embed text or list of texts.

        Args:
            input: A string or list of strings.

        Returns:
            An object with .data[i].embedding for each input text.
        """
        from api.model_types import ModelType

        api_kwargs = self.model_client.convert_inputs_to_api_kwargs(
            input=input,
            model_kwargs=self.model_kwargs,
            model_type=ModelType.EMBEDDER,
        )
        return self.model_client.call(api_kwargs=api_kwargs, model_type=ModelType.EMBEDDER)


def get_embedder(is_local_ollama: bool = False, use_google_embedder: bool = False, embedder_type: str = None) -> Embedder:
    """Get embedder based on configuration or parameters.

    Args:
        is_local_ollama: Legacy parameter for Ollama embedder
        use_google_embedder: Legacy parameter for Google embedder
        embedder_type: Direct specification of embedder type ('ollama', 'google', 'bedrock', 'openai')

    Returns:
        Embedder: Configured embedder instance
    """
    # Determine which embedder config to use
    if embedder_type:
        if embedder_type == 'ollama':
            embedder_config = configs["embedder_ollama"]
        elif embedder_type == 'google':
            embedder_config = configs["embedder_google"]
        elif embedder_type == 'bedrock':
            embedder_config = configs["embedder_bedrock"]
        else:  # default to openai
            embedder_config = configs.get("embedder_direct", configs["embedder"])
    elif is_local_ollama:
        embedder_config = configs["embedder_ollama"]
    elif use_google_embedder:
        embedder_config = configs["embedder_google"]
    else:
        # Auto-detect based on current configuration
        current_type = get_embedder_type()
        if current_type == 'bedrock':
            embedder_config = configs["embedder_bedrock"]
        elif current_type == 'ollama':
            embedder_config = configs["embedder_ollama"]
        elif current_type == 'google':
            embedder_config = configs["embedder_google"]
        else:
            embedder_config = configs.get("embedder_direct", configs["embedder"])

    # Initialize model client
    model_client_class = embedder_config["model_client"]
    if "initialize_kwargs" in embedder_config:
        model_client = model_client_class(**embedder_config["initialize_kwargs"])
    else:
        model_client = model_client_class()

    # Create embedder
    embedder = Embedder(
        model_client=model_client,
        model_kwargs=embedder_config["model_kwargs"],
    )

    # Set batch_size if configured
    if "batch_size" in embedder_config:
        embedder.batch_size = embedder_config["batch_size"]

    return embedder
