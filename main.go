package main

import (
	"fmt"

	"tfidf/model"
	"tfidf/pipeline"
)

func main() {
	// Document Corpus
	docs := []model.Document{
		//{0, "apple orange banana"},
		//{1, "banana apple"},
		//{2, "computer science and data"},
		{0, "1848 N 680 W OREM"},
		{1, "3898 W 5535 S OREM"},
		{2, "3845 W 5400 S SANDY"},
	}

	//Step 1: Preprocessing Data
	vocab, docTFs, df := pipeline.BuildVocabAndTF(docs)
	idf := pipeline.ComputeIDF(df, len(docs))
	docVecs := pipeline.BuildTFIDFVectors(docTFs, idf)

	//Step 2: Indexing (Inverted-Index)
	invertedIndex := pipeline.MakeInvertedIndex(docTFs)

	// Step 3: Query Processing
	query := "3845 S"
	queryVec, candidates := pipeline.BuildQueryTFIDFVector(query, vocab, idf, invertedIndex)

	// Step 4: Scoring
	scores := pipeline.ScoreDocuments(queryVec, docVecs, candidates)

	// Step 5: Returning Results
	for i, score := range scores {
		fmt.Printf("Rank %d: Doc %d (score: %.4f)\n", i+1, score.DocID, score.Value)
	}
}
