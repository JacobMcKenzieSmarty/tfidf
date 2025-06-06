package pipeline

import "strings"

// Tokenize for the purpose of this demo does not do much. You can go crazy with stemming and segmenting, but enforcing lowercase is more than sufficient for this simple case.
func Tokenize(text string) []string {
	return strings.Fields(strings.ToLower(text))
}
