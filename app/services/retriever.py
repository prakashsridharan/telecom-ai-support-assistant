from pathlib import Path
import logging
import math
import re
from collections import Counter

logger = logging.getLogger(__name__)

# Repository root: app/services/retriever.py -> app/services -> app -> root
_PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Stopwords: closed-class function words plus generic conversational verbs.
# With a four-document corpus IDF alone does not suppress these enough — a rare
# generic verb in a short document ("offer escalation" in service_policies.md)
# otherwise outranks the topical term the customer actually asked about.
_STOPWORDS = frozenset(
    """
    a about after all also am an and any are as at be been before but by can could
    did do does for from get give got has have hello help hi how i if in into is it
    its just know let like look me more most much my need no not now of offer offers
    on only or other our out over please provide provides said same say see should so
    some such take tell than thanks that the their them then there these they this
    those to up us use very want was way we well were what when where which who why
    will with work would you your
    """.split()
)


class KnowledgeRetriever:
    """Small transparent lexical retriever for the portfolio demo.

    Scores documents by TF-IDF cosine similarity over stopword-filtered tokens.
    That is still ~40 lines of inspectable arithmetic with no external service,
    but unlike raw term overlap it rewards rare, discriminating terms and
    normalises for document length.

    The interface can later be replaced with a vector database such as
    pgvector, Pinecone, OpenSearch or another embedding store.
    """

    def __init__(self, knowledge_dir: str | Path = "knowledge_base"):
        path = Path(knowledge_dir)
        # Resolve relative paths against the repository root rather than the
        # current working directory, so the app and the tests behave the same
        # regardless of where they are launched from.
        self.knowledge_dir = path if path.is_absolute() else _PROJECT_ROOT / path
        self.documents = self._load_documents()
        self._index()

    def _load_documents(self):
        if not self.knowledge_dir.is_dir():
            logger.error(
                "Knowledge base directory not found",
                extra={"knowledge_dir": str(self.knowledge_dir)},
            )
            return []

        docs = []
        for path in sorted(self.knowledge_dir.glob("*.md")):
            docs.append({"name": path.name, "text": path.read_text(encoding="utf-8")})

        logger.info(
            "Knowledge base loaded",
            extra={
                "knowledge_dir": str(self.knowledge_dir),
                "document_count": len(docs),
            },
        )
        return docs

    def _index(self) -> None:
        """Precompute IDF and a normalised TF-IDF vector per document."""
        total = len(self.documents)
        doc_frequency: Counter[str] = Counter()
        term_counts = []

        for doc in self.documents:
            counts = Counter(self._tokens(doc["text"]))
            term_counts.append(counts)
            doc_frequency.update(counts.keys())

        self._idf = {
            term: math.log((1 + total) / (1 + df)) + 1.0
            for term, df in doc_frequency.items()
        }

        for doc, counts in zip(self.documents, term_counts):
            vector = {
                term: (1 + math.log(count)) * self._idf[term]
                for term, count in counts.items()
            }
            norm = math.sqrt(sum(w * w for w in vector.values())) or 1.0
            doc["vector"] = vector
            doc["norm"] = norm

    @property
    def document_count(self) -> int:
        return len(self.documents)

    @property
    def is_ready(self) -> bool:
        """True when at least one knowledge document is available to retrieve."""
        return bool(self.documents)

    @staticmethod
    def _tokens(text: str) -> list[str]:
        return [
            token
            for token in re.findall(r"[a-z0-9]+", text.lower())
            if token not in _STOPWORDS
        ]

    def search(self, query: str, top_k: int = 3, min_score: float = 0.05):
        """Return the top-k documents above `min_score`, best first.

        `min_score` is what lets an off-topic question retrieve nothing at all,
        so the assistant refuses rather than answering from a weakly-matched
        document.
        """
        query_counts = Counter(self._tokens(query))
        query_vector = {
            term: (1 + math.log(count)) * self._idf[term]
            for term, count in query_counts.items()
            if term in self._idf
        }
        if not query_vector:
            return []

        query_norm = math.sqrt(sum(w * w for w in query_vector.values())) or 1.0

        scored = []
        for doc in self.documents:
            dot = sum(
                weight * doc["vector"].get(term, 0.0)
                for term, weight in query_vector.items()
            )
            if dot <= 0:
                continue
            score = dot / (doc["norm"] * query_norm)
            if score >= min_score:
                scored.append((score, doc))

        scored.sort(key=lambda pair: (-pair[0], pair[1]["name"]))
        return [
            {"source": doc["name"], "content": doc["text"], "score": round(score, 4)}
            for score, doc in scored[:top_k]
        ]
