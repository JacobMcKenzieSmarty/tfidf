package pipeline

import (
	"math"

	"tfidf/model"
)

// BuildVocabAndTermFrequenciesAndDocumentFrequency takes in a slice of raw documents and produces 3 key pieces of output.
// The vocabulary is a lookup structure that can take in a tokenized term and tell you if it exists in the corpus and what it's tokenID is
// For every document in the corpus, a TermFrequencyVector will be created, which can tell you the term frequency (# time term appears in the document / # of all terms in the document)
// of a given tokenID for a given document.  EX: If the document were {A, B, A, C} the TF's would be A: 0.5, B: 0.25, C: 0.25, and 0.0 for all other tokens
// The Document Frequency (DF) if for the corpus as a whole. It will tell you for a given tokenID, how many unique documents it exists in.
// EX: We have documents: {A, B, A, C}, {E}, and {A, B, C, D}
// A's Document Frequency would be 2 since it appears in 2 documents. The DF returned for the corpus would be {A: 2, B: 2, C: 2, D: 1, E: 1}
func BuildVocabAndTermFrequenciesAndDocumentFrequency(docs []model.Document) (model.Vocabulary, []model.TermFrequencyVector, model.DocumentFrequency) {
	vocab := map[string]model.TokenID{}
	df := map[model.TokenID]int{}
	docTFs := make([]model.TermFrequencyVector, len(docs))
	vocabIndex := 0

	for i, doc := range docs {
		tf := model.TermFrequencyVector{}
		seen := map[model.TokenID]bool{}
		for _, token := range Tokenize(doc.Text) {
			id, ok := vocab[token]
			if !ok {
				id = model.TokenID(vocabIndex)
				vocab[token] = id
				vocabIndex++
			}
			tf[id]++
			if !seen[id] {
				df[id]++
				seen[id] = true
			}
		}
		docTFs[i] = tf
	}
	return vocab, docTFs, df
}

// ComputeIDF Takes in the DocumentFrequency "Vector" created from BuildVocabAndTermFrequenciesAndDocumentFrequency and applies the inverse of it.
// What does Inverse Document Frequency (IDF) tell us? It tells us how common/rare a term is across the corpus as a whole (all documents).
// Terms with high IDF scores are considered more important/informative since they are not common across the corpus. It helps pinpoint documents that are specifically
// about certain topics.
// It's main role is to help downweight common terms (think "the", "a", "and") that appear frequently across the corpus but have little informational value.
func ComputeIDF(df model.DocumentFrequency, totalDocs int) model.InverseDocumentFrequencyVector {
	idf := map[model.TokenID]float64{}

	//Why log? This formula is considered "smoothed". It is used in most practical applications of TF-IDF as it better handles edge cases
	// at the cost of placing slightly less emphasis on rare terms.
	for termID, docCount := range df {
		idf[termID] = 1 + math.Log(float64(totalDocs)/float64(1+docCount))
	}
	return idf
}

// ComputeNormalizedTFIDF takes in a document represented as a TermFrequencyVector and the InverseDocumentFrequencyVector (for the corpus of documents as a whole).
// For the provided tf vector, you simply update it by multiplying each term frequency in the vector by the IDF value of the token.
// The resulting TFIDF Vector is a vector of TFIDF features, and we normalize here to make cosine comparison simpler later down the line.
func ComputeNormalizedTFIDF(tf model.TermFrequencyVector, idf model.InverseDocumentFrequencyVector) model.TFIDFVector {
	vec := model.TFIDFVector{}
	var norm float64
	for termID, freq := range tf {
		//Why using log on the term frequency value? It prevents long documents from dominating due to raw counts
		//It also makes very frequent terms have less weight (which makes sense intuitively - words like "a" and "the" hold little distinguishing information.
		tfWeight := 1 + math.Log(freq)
		tfidf := tfWeight * idf[termID]
		vec[termID] = tfidf
		norm += tfidf * tfidf
	}
	norm = math.Sqrt(norm)
	if norm > 0 {
		for termID := range vec {
			vec[termID] /= norm
		}
	}
	return vec
}

// BuildTFIDFVectors is just an aggregator that takes in a slice of termFrequencyVectors and return their corresponding, normalized TFIDF vector
func BuildTFIDFVectors(tfList []model.TermFrequencyVector, idf model.InverseDocumentFrequencyVector) []model.TFIDFVector {
	vectors := make([]model.TFIDFVector, len(tfList))
	for i, tf := range tfList {
		vectors[i] = ComputeNormalizedTFIDF(tf, idf)
	}
	return vectors
}
