# TF-IDF Search Engine in Go

A simple and fast TF-IDF-based text search engine written in Go. It supports tokenization, log-scaled term frequency and inverse document frequency weighting, query vector construction, and cosine similarity ranking.

---

## 🛝 Based off the Presentation

- https://docs.google.com/presentation/d/1ZmHTDNNzgtjNR6vbSmzhrVjvs5qJ-yWKpv9TrPyPbcE/edit?usp=sharing


---

## 🚀 Features

- Tokenizes and indexes a set of short documents
- Computes smoothed log TF-IDF vectors
- Supports vectorized cosine similarity for ranking
- Returns top-k most relevant documents for a query

---

## 🧱 Project Structure

```
tfidf/
├── go.mod              // Module definition
├── main.go             // Entry point
├── model/              // Shared types (Document, Vector, Score)
├── pipeline/           // Tokenizer, TF-IDF logic, search engine
```

---

## 🛠️ Getting Started

### 1. Clone the repo

```bash
git clone https://github.com/JacobMcKenzieSmarty/tfidf.git
cd tfidf
```

### 2. Run the project

```bash
go run main.go
```

---

## 🔍 Example

```go
docs := []model.Document{
    {0, "apple orange banana"},
    {1, "banana apple"},
    {2, "computer science and data"},
}
query := "banana apple"
```

Output:

```
Rank 1: Doc 1 (score: 0.9765)
Rank 2: Doc 0 (score: 0.6123)
Rank 3: Doc 2 (score: 0.0000)
```

---

## 📦 Dependencies

No external libraries — pure Go!

---

## 📜 License

MIT License — feel free to use, modify, and contribute!

---

## 🤝 Contributing

PRs welcome! Open issues or feature requests freely.
