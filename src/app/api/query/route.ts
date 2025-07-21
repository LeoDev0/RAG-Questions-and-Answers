import { NextRequest, NextResponse } from 'next/server';
import { RAGPipeline } from '@/lib/rag-pipeline';

const ragPipeline = new RAGPipeline();

export async function POST(request: NextRequest) {
  try {
    const { question } = await request.json();

    if (!question || typeof question !== 'string') {
      return NextResponse.json(
        { error: 'Question is required and must be a string' },
        { status: 400 }
      );
    }

    const response = await ragPipeline.query(question);

    return NextResponse.json({
      success: true,
      answer: response.answer,
      sources: response.sources,
      confidence: response.confidence,
    });

  } catch (error) {
    console.error('Query error:', error);
    return NextResponse.json(
      { error: 'Failed to process query' },
      { status: 500 }
    );
  }
}