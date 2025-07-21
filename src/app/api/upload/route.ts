import { NextRequest, NextResponse } from 'next/server';
import { RAGPipeline } from '@/lib/rag-pipeline';
import { DocumentProcessor } from '@/lib/document-processor';

const ragPipeline = new RAGPipeline();
const documentProcessor = new DocumentProcessor();

export async function POST(request: NextRequest) {
  try {
    const formData = await request.formData();
    const file = formData.get('file') as File;

    if (!file) {
      return NextResponse.json(
        { error: 'No file provided' },
        { status: 400 }
      );
    }

    // Process the file
    const { content } = await documentProcessor.processFile(file);
    
    // Create document object
    const document = documentProcessor.createDocument(content, file.name);
    
    // Process into chunks
    const chunks = await ragPipeline.processDocument(content, {
      source: file.name,
    });
    
    // Add to vector store
    await ragPipeline.addDocumentToVectorStore(chunks);
    
    // Update document with chunks
    document.chunks = chunks;

    return NextResponse.json({
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
    return NextResponse.json(
      { error: 'Failed to process document' },
      { status: 500 }
    );
  }
}