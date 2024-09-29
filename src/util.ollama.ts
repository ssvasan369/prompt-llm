import { ChromaClient, OllamaEmbeddingFunction } from 'chromadb';
import { OLLAMA_MODEL_EMBEDDING, NAME_EMBEDDING_COLLECTION, OLLAMA_MODEL } from './constants';
import { ChatResponse, Ollama, Tool } from 'ollama';
import { v4 as uuidv4 } from 'uuid';

const ollama = new Ollama({ host: 'http://rag_ollama:11434' });
const clientChromaDB = new ChromaClient({
    path: process.env.CHROMADB_PATH!,
});

const embedder = new OllamaEmbeddingFunction({
    url: `http://rag_ollama:11434/api/embeddings`,
    model: OLLAMA_MODEL_EMBEDDING!,
});

/**
 * Generate a prompt by combining a user prompt and provided data.
 * @param prompt - The original prompt from the user.
 * @param data - Data to use within the prompt (either a string or number array).
 * @returns Generated prompt string.
 */
function generatePrompt(prompt: string, data: string | number[]): string {
    return `Using this data: ${data}. Respond to this prompt: ${prompt}`;
}

/**
 * Get a chat response from Ollama's model.
 * @param model - The model to use for chat.
 * @param messages - Array of role-content messages.
 * @param tools - Array of tools to assist in chat.
 * @returns The chat response.
 */
export const getOllamaChatResponse = async (
    model: string,
    messages: { role: string; content: string }[],
    tools: Tool[],
): Promise<ChatResponse | undefined> => {
    try {
        const response = await ollama.chat({
            model,
            messages,
            stream: false,
            tools,
        });
        return response;
    } catch (error) {
        console.error('Error getting Ollama chat response:', error);
        return undefined;
    }
};

/**
 * Initialize Ollama embeddings by adding documents to a ChromaDB collection.
 * @param documents - Array of documents to embed.
 */
export async function initOllamaEmbedding(documents: string[]): Promise<void> {
    try {
        const collection = await clientChromaDB.createCollection({
            name: NAME_EMBEDDING_COLLECTION,
            embeddingFunction: embedder,
        });

        const addDocumentPromises = documents.map(async (document) => {
            const uniqueId = uuidv4();
            await collection.add({
                ids: [uniqueId],
                documents: [document],
                metadatas: [{ name: document }],
            });
        });

        await Promise.all(addDocumentPromises);
    } catch (error) {
        console.error('Error initializing Ollama embeddings:', error);
    }
}

/**
 * Retrieve Ollama embeddings and query them from the collection.
 * @param prompt - The user prompt for embedding retrieval.
 * @param documents - Array of documents to embed.
 * @returns The best matching document based on the query or an empty string.
 */
export async function getOllamaEmbeddingRetrieve(prompt: string, documents: string[]): Promise<string> {
    console.log(prompt, documents);
    
    try {

        let collection = await clientChromaDB.getCollection({
            name: NAME_EMBEDDING_COLLECTION,
            embeddingFunction: embedder,
        });

        if (!collection) {
            await initOllamaEmbedding(documents);
            collection = await clientChromaDB.getCollection({
                name: NAME_EMBEDDING_COLLECTION,
                embeddingFunction: embedder,
            });
        }

        const response = await ollama.embeddings({
            model: OLLAMA_MODEL_EMBEDDING,
            prompt,
        });

        const queryResults = await collection.query({
            queryEmbeddings: [response.embedding],
            nResults: 1,
        });

        return queryResults.documents?.[0]?.[0] ?? '';
    } catch (error) {
        console.error('Error retrieving Ollama embedding:', error);
        return '';
    }
}

/**
 * Generate a human-readable response by combining an original question and embeddings.
 * @param originalQuestion - The user's original question.
 * @param embeddings - The embeddings to include in the prompt.
 * @returns A chat response or undefined if an error occurs.
 */
export async function getOllamaHumanResponse(
    originalQuestion: string,
    embeddings: string
): Promise<ChatResponse | undefined> {
    try {
        const question = generatePrompt(originalQuestion, embeddings);
        const response = await getOllamaChatResponse(
            OLLAMA_MODEL,
            [{ role: 'user', content: question }],
            []
        );
        return response;
    } catch (error) {
        console.error('Error getting Ollama human response:', error);
        return undefined;
    }
}
