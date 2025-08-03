package main

import (
	"log"
	"rag-backend/internal/repositories/vectorstore/memory"

	"github.com/gin-gonic/gin"
	"github.com/rs/cors"

	"rag-backend/internal/config"
	"rag-backend/internal/handlers"
	"rag-backend/internal/services"
)

func main() {
	cfg := config.Load()

	vectorStore := memory.NewMemoryVectorStore()

	ragPipeline := services.NewRAGPipeline(cfg, vectorStore)
	documentProcessor := services.NewDocumentProcessor()

	uploadHandler := handlers.NewUploadHandler(ragPipeline, documentProcessor)
	queryHandler := handlers.NewQueryHandler(ragPipeline)
	healthHandler := handlers.NewHealthHandler()

	router := gin.Default()

	c := cors.New(cors.Options{
		AllowedOrigins:   []string{"http://localhost:3000", "http://127.0.0.1:3000"},
		AllowedMethods:   []string{"GET", "POST", "PUT", "DELETE", "OPTIONS"},
		AllowedHeaders:   []string{"*"},
		AllowCredentials: true,
	})
	router.Use(func(ctx *gin.Context) {
		c.HandlerFunc(ctx.Writer, ctx.Request)
		ctx.Next()
	})

	// File upload size limit (10MB)
	router.MaxMultipartMemory = 10 << 20 // 10MB

	api := router.Group("/api")
	{
		api.POST("/upload", uploadHandler.HandleUpload)
		api.POST("/query", queryHandler.HandleQuery)
	}

	router.GET("/health", healthHandler.HandleHealth)

	log.Printf("🚀 Backend server starting on port %s", cfg.Port)
	if err := router.Run(":" + cfg.Port); err != nil {
		log.Fatal("Failed to start server:", err)
	}
}
