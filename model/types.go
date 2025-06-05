package model

type Document struct {
	ID   int
	Text string
}

type Vector map[int]float64

type Score struct {
	DocID int
	Value float64
}
