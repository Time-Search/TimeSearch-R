def is_conversational(maybe_messages) -> bool:
    r"""
    Check if the example is in a conversational format.

    Args:
        example (`dict[str, Any]`):
            A single data entry of a dataset. The example can have different keys depending on the
            dataset type.

    Returns:
        `bool`:
            `True` if the data is in a conversational format, `False` otherwise.

    Examples:

    ```python
    >>> example = [{"role": "user", "content": "What color is the sky?"}]
    >>> is_conversational(example)
    True
    >>> example = "The sky is"
    >>> is_conversational(example)
    False
    ```
    """
    # It must be a list of messages,
    if isinstance(maybe_messages, list):
        maybe_message = maybe_messages[0]
        # Each message must a list of dictionaries with keys "role" and "content"
        if isinstance(maybe_message, dict) and "role" in maybe_message and "content" in maybe_message:
            return True
    return False


def extract_completion(maybe_messages) -> str:
    if is_conversational(maybe_messages):
        return maybe_messages[-1]["content"]
    else:
        return maybe_messages

def extract_completion_batch(maybe_messages_batch):
    return [extract_completion(maybe_messages) for maybe_messages in maybe_messages_batch]
