from collections.abc import Iterator

import tiktoken

_ENCODER = tiktoken.get_encoding("cl100k_base")


def split_into_chunks(
    text: str,
    max_tokens: int = 800,
    overlap_tokens: int = 100,
) -> list[str]:
    """
    Split text into token-aware chunks with a sliding overlap window.
    Overlap preserves context at chunk boundaries for better entity extraction.
    """
    tokens = _ENCODER.encode(text)
    if not tokens:
        return []

    chunks: list[str] = []
    start = 0
    while start < len(tokens):
        end = min(start + max_tokens, len(tokens))
        chunks.append(_ENCODER.decode(tokens[start:end]))
        if end == len(tokens):
            break
        start += max_tokens - overlap_tokens

    return chunks


def iter_chunks(
    text: str,
    max_tokens: int = 800,
    overlap_tokens: int = 100,
) -> Iterator[str]:
    """Yield chunks one at a time to keep memory usage flat for large documents."""
    yield from split_into_chunks(text, max_tokens, overlap_tokens)


def count_tokens(text: str) -> int:
    """Return the token count for a string using the cl100k_base encoder."""
    return len(_ENCODER.encode(text))
