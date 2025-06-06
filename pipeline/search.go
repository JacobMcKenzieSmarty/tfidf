package pipeline

import (
	"sort"

	"tfidf/model"
)

// BuildQueryTFIDFVector takes in the query string and prepares it just like we did with the documents from the corpus. We treat it as a document and create it a
// normalized TFIDF vector. The function also takes in the invertedIndex which allows us to grab all the documents from the corpus that at least contain some of the
// same terms as the query - candidates (cuts down our search space, i.e. the number of vectors we have to compare and calculate similarity scores for)
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

// ScoreDocuments goes calculates how similar the constructed, normalized TFIDF query vector is to the candidate normalized TFIDF vectors from the corpus.
// It returns scores in descending order (highest to lowest)
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

// calculateCosineSimilarity lets us know how similar the query vector is to the given documentVector (from the corpus).
// Scores range from 0 to 1 (inclusive) where scores closer to 1 indicate higher similarity.
func calculateCosineSimilarity(queryVec, docVec model.TFIDFVector) float64 {
	var score float64                // 0 ≤ score ≤ 1   where 0 means totally dissimilar and 1 meaning perfectly similar
	for id, qVal := range queryVec { //here is the dot product calculation for the candidates, which is the cos(𝜃)
		if dVal, ok := docVec[id]; ok {
			score += qVal * dVal
		}
	}
	return score
}
