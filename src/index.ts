import express, { Request, Response } from 'express';
import { config } from 'dotenv';
import { MODEL_DOCUMENTS, MAIN_PROMPT } from './constants';
import { getOllamaEmbeddingRetrieve, getOllamaHumanResponse } from './util.ollama';
import { ChatResponse } from 'ollama';

// Load environment variables
require('dotenv').config()

// Initialize Express server
const app = express();
const PORT = process.env.PORT || 3000;

/**
 * Starts the RAG bot process by getting embeddings and generating a response.
 * @param prompt - The user-provided prompt.
 * @returns The human-readable response from Ollama.
 */
async function startRAGBot(prompt: string): Promise<string | undefined> {
    const cleanedPrompt = prompt.trim();

    // Get embedding and use it to generate a response
    const embedding = await getOllamaEmbeddingRetrieve(cleanedPrompt, MODEL_DOCUMENTS);
    const response: ChatResponse | undefined = await getOllamaHumanResponse(cleanedPrompt, embedding);

    // Ensure the response is a string or undefined
    return response?.message.content ?? undefined;  // Extract 'content' from 'ChatResponse'
}

// Route to handle RAG bot responses
app.get('/ragbot', async (req: Request, res: Response) => {
    try {
        const userPrompt = (req.query.prompt as string) || MAIN_PROMPT;
        console.log('User Prompt:', userPrompt);

        // Start the RAG bot and get the response
        const response = await startRAGBot(userPrompt);

        // Respond to the user
        res.json({
            success: true,
            prompt: userPrompt,
            response,
        });
    } catch (error) {
        // Type narrowing to ensure 'error' is an instance of Error
        if (error instanceof Error) {
            console.error('Error starting RAG Bot:', error.message);
            res.status(500).json({
                success: false,
                message: 'Internal Server Error',
                error: error.message,
            });
        } else {
            // Handle non-Error objects safely
            console.error('Unknown error:', error);
            res.status(500).json({
                success: false,
                message: 'An unknown error occurred',
            });
        }
    }
});

// Start server listening on the specified port
app.listen(PORT, () => {
    console.log(`Server is running on port ${PORT}`);
});
