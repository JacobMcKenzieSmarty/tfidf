package pipeline

import "tfidf/model"

type InvertedIndex map[model.TokenID]map[model.DocID]struct{}

func MakeInvertedIndex(docTFs []model.TermFrequencyVector) InvertedIndex {
	invertedIndex := InvertedIndex{}
	for docID, tf := range docTFs {
		for tokenID := range tf {
			if invertedIndex[tokenID] == nil {
				invertedIndex[tokenID] = make(map[model.DocID]struct{})
			}
			invertedIndex[tokenID][model.DocID(docID)] = struct{}{}
		}
	}
	return invertedIndex
}
