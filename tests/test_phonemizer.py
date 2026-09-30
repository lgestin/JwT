import threading
import time
from concurrent.futures import ThreadPoolExecutor

from misaki.token import MToken

from jwt.data.text.phonemizer import Phonemizer


def test_concurrent_phonemize_calls_never_overlap() -> None:
    """Threads take turns in G2P, since its espeak fallback is not thread-safe."""
    phonemizer = Phonemizer.__new__(Phonemizer)
    phonemizer.lock = threading.Lock()
    active, overlaps = 0, 0

    def g2p(text: str) -> tuple[str, list[MToken]]:
        nonlocal active, overlaps
        active += 1
        overlaps += active > 1
        time.sleep(0.001)  # widen the window a concurrent call would hit
        active -= 1
        return text, []

    phonemizer.g2p = g2p  # ty: ignore[invalid-assignment]
    with ThreadPoolExecutor(8) as pool:
        out = list(pool.map(phonemizer.phonemize, [str(i) for i in range(200)]))

    assert overlaps == 0
    assert [ph for ph, _ in out] == [str(i) for i in range(200)]
