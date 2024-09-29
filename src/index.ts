import { Ollama } from 'ollama';
import prompts from 'prompts';

const OLLAMA_HOST = 'http://127.0.0.1:11434';
const MODEL_NAME = 'llava:latest';

async function main() {
    const ollama = new Ollama({ host: OLLAMA_HOST });

    console.log('Q & A With AI');
    console.log('=============');
    console.log('Type /bye to stop the program');
    let exit = false;

    while (!exit) {
        const { question } = await prompts({
            type: 'text',
            name: 'question',
            message: 'Your question: ',
            validate: value => (value ? true : 'Question cannot be empty'),
        });
        if (question == '/bye') {
            console.log('See you later!');
            exit = true;
        } else {
            const response = await ollama.generate({
                model: MODEL_NAME,
                prompt: question,
                stream: true
            });
            for await (const chunk of response) {
                console.log(chunk.response);
                
            }
        }
    }
}


// Execute the main function
main().catch((error) => {
    console.error("An unexpected error occurred:", error);
});