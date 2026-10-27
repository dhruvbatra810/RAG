import re


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
    """
    Split text into overlapping chunks that respect sentence boundaries.

    Args:
        text:       The input text to chunk.
        chunk_size: Max number of words per chunk.
        overlap:    Number of words to carry over into the next chunk for context.

    Returns:
        A list of text chunks.
    """
    # Step 1: Split into sentences on punctuation boundaries
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())

    chunks: list[str] = []
    current: list[str] = []   # sentences in the current chunk
    word_count: int = 0        # running word count for current chunk

    for sentence in sentences:
        sentence_word_count = len(sentence.split())

        # Step 2: If adding this sentence exceeds the chunk size, save the chunk
        if word_count + sentence_word_count > chunk_size and current:
            chunks.append(" ".join(current))

            # Step 3: Carry over the last `overlap` words for context continuity
            overlap_text = " ".join(current).split()
            overlap_words = overlap_text[-overlap:] if overlap else []
            current = [" ".join(overlap_words)] if overlap_words else []
            word_count = len(overlap_words)

        current.append(sentence)
        word_count += sentence_word_count

    # Save the last remaining chunk
    if current:
        chunks.append(" ".join(current))

    return chunks
