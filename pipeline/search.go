package pipeline

import (
	"sort"

	"tfidf/model"
)

func BuildQueryVector(query string, vocab map[string]int, idf map[int]float64) model.Vector {
	tf := model.Vector{}
	for _, token := range Tokenize(query) {
		if id, ok := vocab[token]; ok {
			tf[id]++
		}
	}
	return ComputeNormalizedTFIDF(tf, idf)
}

func ScoreDocuments(queryVec model.Vector, docVecs []model.Vector) []model.Score {
	scores := []model.Score{}

	for i, docVec := range docVecs {
		var score float64
		for id, qval := range queryVec {
			if dval, ok := docVec[id]; ok {
				score += qval * dval
			}
		}
		scores = append(scores, model.Score{DocID: i, Value: score})
	}

	sort.Slice(scores, func(i, j int) bool {
		return scores[i].Value > scores[j].Value
	})

	return scores
}
