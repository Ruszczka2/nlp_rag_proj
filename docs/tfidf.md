# TF-IDF

TF-IDF (Term Frequency–Inverse Document Frequency) to numeryczna reprezentacja tekstu, która waży słowa według tego, jak często występują w dokumencie i jak rzadkie są w całym korpusie. Słowa częste w jednym dokumencie, ale rzadkie ogólnie, otrzymują wyższe wyniki. W scikit-learn `TfidfVectorizer` zamienia surowy tekst na rzadkie wektory nadające się do klasyfikatorów takich jak LinearSVC lub regresja logistyczna.
