package data

import (
	"encoding/csv"
	"fmt"
	"os"
	"strconv"

	"tfidf/model"
)

func LoadDocumentsFromCSV(path string) ([]model.Document, error) {
	file, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("error opening CSV: %w", err)
	}
	defer file.Close()

	reader := csv.NewReader(file)
	records, err := reader.ReadAll()
	if err != nil {
		return nil, fmt.Errorf("error reading CSV: %w", err)
	}

	var docs []model.Document
	for i, row := range records {
		if i == 0 {
			continue // skip header
		}
		id, err := strconv.Atoi(row[0])
		if err != nil {
			return nil, fmt.Errorf("invalid doc_id at row %d: %w", i, err)
		}
		category := row[1]
		text := row[2] // row[1] is "category", row[2] is "text"
		docs = append(docs, model.Document{
			ID:       id,
			Text:     text,
			Category: category,
		})
	}
	return docs, nil
}
