package pipeline

import "tfidf/model"

type InvertedIndex map[model.TokenID]map[model.DocID]struct{}

// MakeInvertedIndex provides you with a structure that lets you find all documents that contain a given TokenID, which is super helpful for limiting the search space.
// Note that this example only gives you back the document ID's. In real world applications, you could also give back Documents AND the TFIDF scores for the given TokenID query.
// You could then use further employ heuristics based on TFIDF to reduce the search space even further. (i.e. only take ones above certain TFIDF threshold)
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
