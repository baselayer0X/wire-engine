"""Split posts into retrievable passages. A whole post embeds to one blurry
average; chunks let retrieval reach the exact passage that answers a question."""


def chunk_text(text, max_chars=900):
    """Greedy paragraph-packing up to ~max_chars. Hard-splits oversized paras."""
    paras = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks, current = [], ""
    for p in paras:
        if len(current) + len(p) + 2 <= max_chars:
            current = (current + "\n\n" + p).strip()
        else:
            if current:
                chunks.append(current)
                current = ""
            if len(p) > max_chars:
                for i in range(0, len(p), max_chars):
                    chunks.append(p[i:i + max_chars])
            else:
                current = p
    if current:
        chunks.append(current)
    return chunks


def chunk_all(posts):
    """posts: list of {title, url, text} -> list of {title, url, text} chunks."""
    out = []
    for post in posts:
        for piece in chunk_text(post["text"]):
            out.append({
                "title": post["title"],
                "url": post.get("url", ""),
                "text": piece,
            })
    return out
