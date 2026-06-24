# RAG (Retrieval-Augmented Generation)

RAG łączy wyszukiwanie informacji z generowaniem tekstu: system najpierw przeszukuje zbiór dokumentów w poszukiwaniu istotnych fragmentów, a następnie przekazuje je jako kontekst do modelu językowego. To ogranicza halucynacje, bo model odpowiada na podstawie znalezionego tekstu, a nie wyłącznie pamięci z treningu. Typowy pipeline: podział dokumentów na chunki, embeddingi, wyszukanie najbliższych dopasowań do pytania, generacja odpowiedzi z pobranego kontekstu.
