package main

import (
	"fmt"

	"tfidf/model"
	"tfidf/pipeline"
)

func main() {
	// Step 1: Input
	docs := []model.Document{
		//{0, "apple orange banana"},
		//{1, "banana apple"},
		//{2, "computer science and data"},
		{0, "1848 N 680 W OREM"},
		{1, "3898 W 5535 S OREM"},
		{2, "3845 W 5400 S SANDY"},
	}

	vocab, docTFs, df := pipeline.BuildVocabAndTF(docs)
	idf := pipeline.ComputeIDF(df, len(docs))

	// Step 2: Document vectors
	docVecs := pipeline.BuildTFIDFVectors(docTFs, idf)
	invertedIndex := pipeline.MakeInvertedIndex(docTFs)

	// Step 3: Query vector
	query := "3845 S"
	queryVec, candidates := pipeline.BuildQueryTFIDFVector(query, vocab, idf, invertedIndex)

	// Step 4: Score
	scores := pipeline.ScoreDocuments(queryVec, docVecs, candidates)

	// Step 5: Print
	for i, score := range scores {
		fmt.Printf("Rank %d: Doc %d (score: %.4f)\n", i+1, score.DocID, score.Value)
	}
}
