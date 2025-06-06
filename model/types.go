package model

type Document struct {
	ID       int
	Text     string
	Category string
}

type TokenID int
type DocID int

type TFIDFVector map[TokenID]float64

type Score struct {
	DocID DocID
	Value float64
}

type Vocabulary map[string]TokenID
type DocumentFrequency map[TokenID]int
type TermFrequencyVector map[TokenID]float64
type InverseDocumentFrequencyVector map[TokenID]float64
