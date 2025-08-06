package codes

// Upload error codes
const (
	ErrNoFile       = "NO_FILE"
	ErrFileTooLarge = "FILE_TOO_LARGE"
	ErrProcessing   = "PROCESSING_ERROR"
	ErrChunking     = "CHUNKING_ERROR"
	ErrStorage      = "STORAGE_ERROR"
)

// Query error codes
const (
	ErrInvalidRequest = "INVALID_REQUEST"
	ErrEmptyQuestion  = "EMPTY_QUESTION"
	ErrQueryError     = "QUERY_ERROR"
)
