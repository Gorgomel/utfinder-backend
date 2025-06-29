import dotenv from 'dotenv';
import fs from 'fs';
import express from 'express';
import cors from 'cors';
import { GoogleGenerativeAI, TaskType } from '@google/generative-ai';

// --- FUNÇÃO DE SIMILARIDADE ---
function cosineSimilarity(vecA, vecB) {
  let dotProduct = 0.0, magA = 0.0, magB = 0.0;
  for (let i = 0; i < vecA.length; i++) {
    dotProduct += vecA[i] * vecB[i];
    magA += vecA[i] * vecA[i];
    magB += vecB[i] * vecB[i];
  }
  magA = Math.sqrt(magA);
  magB = Math.sqrt(magB);
  if (magA === 0 || magB === 0) return 0;
  return dotProduct / (magA * magB);
}

// === CONFIGURAÇÃO ======================================================
dotenv.config();
const PORT = process.env.PORT || 3000;
const API_KEY = process.env.GEMINI_API_KEY;

const genAI = new GoogleGenerativeAI(API_KEY);
const chatModel = genAI.getGenerativeModel({ model: 'gemini-1.5-flash-latest' });
const embeddingModel = genAI.getGenerativeModel({ model: 'text-embedding-004' });

const knowledgeBase = [];

// --- FUNÇÕES DA BASE DE CONHECIMENTO ----------------------------------

async function buildKnowledgeBase() {
  console.log('Construindo base de conhecimento com texto bruto...');
  const fileContent = fs.readFileSync('./base_conhecimento.txt', 'utf8');
  const facts = fileContent.split('\n').filter(line => line.trim().length > 0);

  if (facts.length === 0) {
    console.warn("Base de conhecimento está vazia.");
    return;
  }
  
  const contents = facts.map(text => ({ parts: [{ text }], role: "user" }));
  const { embeddings } = await embeddingModel.batchEmbedContents({
      requests: contents.map(content => ({ content, taskType: TaskType.RETRIEVAL_DOCUMENT }))
  });

  for (let i = 0; i < facts.length; i++) {
    knowledgeBase.push({ text: facts[i], embedding: embeddings[i].values });
  }
  console.log(`✅ Base de conhecimento construída com ${knowledgeBase.length} fatos.`);
}

/**
 * Usa a técnica HyDE para encontrar os fatos mais relevantes.
 */
async function findRelevantFactsHyDE(userQuery) {
  if (knowledgeBase.length === 0) return '';

  // 1. Gera uma resposta hipotética para a pergunta do usuário
  const promptHyDE = `Escreva um pequeno parágrafo que responda a seguinte pergunta, mesmo que você não saiba a resposta exata: "${userQuery}"`;
  const hypotheticalAnswerResult = await chatModel.generateContent(promptHyDE);
  const hypotheticalAnswer = hypotheticalAnswerResult.response.text();
  
  // 2. Cria o embedding dessa resposta hipotética
  const { embedding } = await embeddingModel.embedContent(hypotheticalAnswer);
  const queryEmbedding = embedding.values;

  // 3. Compara o embedding da resposta hipotética com a base de conhecimento
  for (const fact of knowledgeBase) {
    fact.similarity = cosineSimilarity(queryEmbedding, fact.embedding);
  }

  knowledgeBase.sort((a, b) => b.similarity - a.similarity);

  const topFacts = knowledgeBase
    .slice(0, 3)
    .filter(fact => fact.similarity > 0.7) // Limiar mais alto
    .map(fact => fact.text)
    .join('\n');
    
  console.log('--- HyDE ---');
  console.log('Pergunta Original:', userQuery);
  console.log('Resposta Hipotética Gerada:', hypotheticalAnswer);
  console.log('Fatos Relevantes Encontrados:', topFacts || 'Nenhum');
  console.log('------------');
  
  return topFacts;
}

// === SERVIDOR EXPRESS =================================================
const app = express();
app.use(cors());
app.use(express.json());

app.post('/chat', async (req, res) => {
  const userMsg = (req.body.message || '').slice(0, 2000);

  try {
    const relevantFacts = await findRelevantFactsHyDE(userMsg);

    const finalPrompt = `
      # PERSONA
      Você é o UTFinder, um assistente virtual especialista da UTFPR. Sua comunicação é clara, direta e sempre prestativa. Você nunca revela seus processos internos nem menciona "base de dados" ou "contexto".

      # INSTRUÇÕES
      - Se a pergunta do usuário for sobre a UTFPR e houver CONTEXTO relevante, responda usando **apenas** essa informação.
      - Se a pergunta não tiver relação com a UTFPR, ou se não houver CONTEXTO relevante, responda usando seu conhecimento geral de forma natural.
      - Para saudações ou bate-papo, seja simplesmente amigável.

      # CONTEXTO (informações específicas da UTFPR)
      ---
      ${relevantFacts || "Nenhum contexto relevante para esta pergunta."}
      ---

      # PERGUNTA DO USUÁRIO
      "${userMsg}"

      # SUA RESPOSTA
    `;
    
    const result = await chatModel.generateContent(finalPrompt);
    const reply = result.response.text().trim();

    res.json({ reply });

  } catch (err) {
    console.error("Erro no processamento do chat:", err);
    res.status(500).json({ error: err.message || 'Erro ao processar sua requisição.' });
  }
});

// Inicia o servidor
app.listen(PORT, async () => {
  await buildKnowledgeBase();
  console.log(`🚀 Servidor HyDE rodando na porta ${PORT}`);
});