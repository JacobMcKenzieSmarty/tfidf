package pipeline

import (
	"sort"

	"tfidf/model"
)

func BuildQueryTFIDFVector(query string, vocab model.Vocabulary, idf model.InverseDocumentFrequencyVector, index InvertedIndex) (model.TFIDFVector, map[model.DocID]struct{}) {
	tf := model.TermFrequencyVector{}
	candidates := make(map[model.DocID]struct{})

	for _, token := range Tokenize(query) {
		if id, ok := vocab[token]; ok {
			tf[id]++

			for docID := range index[id] {
				candidates[docID] = struct{}{}
			}
		}
	}
	return ComputeNormalizedTFIDF(tf, idf), candidates
}

func ScoreDocuments(queryVec model.TFIDFVector, docVecs []model.TFIDFVector, candidates map[model.DocID]struct{}) []model.Score {
	var scores []model.Score

	for docID := range candidates {
		scores = append(scores, model.Score{DocID: docID, Value: calculateCosineSimilarity(queryVec, docVecs[docID])})
	}

	sort.Slice(scores, func(i, j int) bool {
		return scores[i].Value > scores[j].Value
	})

	return scores
}

func calculateCosineSimilarity(queryVec, docVec model.TFIDFVector) float64 {
	var score float64                // 0 ≤ score ≤ 1   where 0 means totally dissimilar and 1 meaning perfectly similar
	for id, qVal := range queryVec { //here is the dot product calculation for the candidates, which is the cos(𝜃)
		if dVal, ok := docVec[id]; ok {
			score += qVal * dVal
		}
	}
}
