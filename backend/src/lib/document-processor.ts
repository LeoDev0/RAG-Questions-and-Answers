import pdf from 'pdf-parse';
import { Document } from '@/types';

export class DocumentProcessor {
  async processPDF(buffer: Buffer): Promise<string> {
    try {
      const data = await pdf(buffer);
      return data.text;
    } catch (error) {
      throw new Error(`Failed to process PDF: ${error}`);
    }
  }

  async processFile(file: Express.Multer.File): Promise<{ content: string; type: string }> {
    const buffer = file.buffer;
    const fileType = file.mimetype;

    if (fileType === 'application/pdf') {
      const content = await this.processPDF(buffer);
      return { content, type: 'pdf' };
    } else if (fileType === 'text/plain') {
      const content = buffer.toString('utf-8');
      return { content, type: 'text' };
    } else {
      throw new Error(`Unsupported file type: ${fileType}`);
    }
  }

  createDocument(
    content: string,
    fileName: string,
  ): Document {
    return {
      id: `${Date.now()}-${Math.random().toString(36).substr(2, 9)}`,
      name: fileName,
      content,
      chunks: [],
      uploadedAt: new Date(),
    };
  }
}