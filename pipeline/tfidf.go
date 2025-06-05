package pipeline

import (
	"math"

	"tfidf/model"
)

func BuildVocabAndTF(docs []model.Document) (map[string]int, []model.Vector, map[int]int) {
	vocab := map[string]int{}
	df := map[int]int{}
	docTFs := make([]model.Vector, len(docs))
	vocabIndex := 0

	for i, doc := range docs {
		tf := model.Vector{}
		seen := map[int]bool{}
		for _, token := range Tokenize(doc.Text) {
			id, ok := vocab[token]
			if !ok {
				id = vocabIndex
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

func ComputeIDF(df map[int]int, totalDocs int) map[int]float64 {
	idf := map[int]float64{}
	for termID, docCount := range df {
		idf[termID] = 1 + math.Log(float64(totalDocs)/float64(1+docCount))
	}
	return idf
}

func ComputeNormalizedTFIDF(tf map[int]float64, idf map[int]float64) model.Vector {
	vec := model.Vector{}
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

func BuildTFIDFVectors(tfList []model.Vector, idf map[int]float64) []model.Vector {
	vectors := make([]model.Vector, len(tfList))
	for i, tf := range tfList {
		vectors[i] = ComputeNormalizedTFIDF(tf, idf)
	}
	return vectors
}
