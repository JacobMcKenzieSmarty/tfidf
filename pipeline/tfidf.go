package pipeline

import (
	"math"

	"tfidf/model"
)

func BuildVocabAndTF(docs []model.Document) (model.Vocabulary, []model.TermFrequencyVector, model.DocumentFrequency) {
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

func ComputeIDF(df model.DocumentFrequency, totalDocs int) model.InverseDocumentFrequencyVector {
	idf := map[model.TokenID]float64{}
	for termID, docCount := range df {
		idf[termID] = 1 + math.Log(float64(totalDocs)/float64(1+docCount))
	}
	return idf
}

func ComputeNormalizedTFIDF(tf model.TermFrequencyVector, idf model.InverseDocumentFrequencyVector) model.TFIDFVector {
	vec := model.TFIDFVector{}
	var norm float64
	for termID, freq := range tf {
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

func BuildTFIDFVectors(tfList []model.TermFrequencyVector, idf model.InverseDocumentFrequencyVector) []model.TFIDFVector {
	vectors := make([]model.TFIDFVector, len(tfList))
	for i, tf := range tfList {
		vectors[i] = ComputeNormalizedTFIDF(tf, idf)
	}
	return vectors
}
