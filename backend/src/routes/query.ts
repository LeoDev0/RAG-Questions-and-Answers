import { Router, Request, Response } from 'express';
import { RAGPipeline } from '@/lib/rag-pipeline';
import { QueryResponse } from '@/types';

const router = Router();
const ragPipeline = new RAGPipeline();

interface QueryRequest {
  question: string;
}

router.post('/', async (req: Request<{}, QueryResponse, QueryRequest>, res: Response<QueryResponse>) => {
  try {
    const { question } = req.body;

    if (!question) {
      return res.status(400).json({
        success: false,
        error: 'Question is required and must be a string'
      });
    }

    const response = await ragPipeline.query(question);

    res.json({
      success: true,
      answer: response.answer,
      sources: response.sources,
      confidence: response.confidence,
    });

  } catch (error) {
    console.error('Query error:', error);
    res.status(500).json({
      success: false,
      error: 'Failed to process query'
    });
  }
});

export default router;