import { Router, Request, Response } from 'express';
import { RAGPipeline } from '@/lib/rag-pipeline';
import { DocumentProcessor } from '@/lib/document-processor';
import { upload } from '@/middleware/upload';
import { UploadResponse } from '@/types';

const router = Router();
const ragPipeline = new RAGPipeline();
const documentProcessor = new DocumentProcessor();

router.post('/', upload.single('file'), async (req: Request, res: Response<UploadResponse>) => {
  try {
    const file = req.file;

    if (!file) {
      return res.status(400).json({
        success: false,
        error: 'No file provided'
      });
    }

    // Process the file
    const { content } = await documentProcessor.processFile(file);
    
    // Create document object
    const document = documentProcessor.createDocument(content, file.originalname);
    
    // Process into chunks
    const chunks = await ragPipeline.processDocument(content, {
      source: file.originalname,
    });
    
    // Add to vector store
    await ragPipeline.addDocumentToVectorStore(chunks);
    
    // Update document with chunks
    document.chunks = chunks;

    res.json({
      success: true,
      document: {
        id: document.id,
        name: document.name,
        chunksCount: chunks.length,
        uploadedAt: document.uploadedAt,
      },
    });

  } catch (error) {
    console.error('Upload error:', error);
    res.status(500).json({
      success: false,
      error: 'Failed to process document'
    });
  }
});

export default router;