// server.js - Versão FINAL com Conhecimento Híbrido e Priorização

import dotenv from 'dotenv';
import fs from 'fs';
import express from 'express';
import cors from 'cors';
import { GoogleGenerativeAI, TaskType } from '@google/generative-ai';

// --- FUNÇÃO DE SIMILARIDADE ---
function cosineSimilarity(vecA, vecB) {
  let dotProduct = 0.0;
  let magA = 0.0;
  let magB = 0.0;
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
  console.log('Iniciando construção da base de conhecimento...');
  const fileContent = fs.readFileSync('./base_conhecimento.txt', 'utf8');
  const qaPairs = fileContent.split('\n\n').filter(p => p.trim());

  if (qaPairs.length === 0) {
    console.warn("Base de conhecimento está vazia.");
    return;
  }

  const requests = qaPairs.map(pair => ({
    content: { parts: [{ text: pair }], role: "user" },
    taskType: TaskType.RETRIEVAL_DOCUMENT,
    title: pair.substring(0, pair.indexOf('\n'))
  }));

  const { embeddings } = await embeddingModel.batchEmbedContents({ requests });

  for (let i = 0; i < qaPairs.length; i++) {
    knowledgeBase.push({
      text: qaPairs[i],
      embedding: embeddings[i].values,
    });
  }
  console.log(`✅ Base de conhecimento construída com ${knowledgeBase.length} pares.`);
}

async function findRelevantFacts(userQuery) {
  if (knowledgeBase.length === 0) return '';
  
  const { embedding } = await embeddingModel.embedContent({
    content: { parts: [{ text: userQuery }], role: "user" },
    taskType: TaskType.RETRIEVAL_QUERY
  });
  const queryEmbedding = embedding.values;

  for (const fact of knowledgeBase) {
    fact.similarity = cosineSimilarity(queryEmbedding, fact.embedding);
  }

  knowledgeBase.sort((a, b) => b.similarity - a.similarity);

  // Aumentamos para 4 para dar mais contexto, mas com um limiar de relevância
  return knowledgeBase
    .slice(0, 4)
    .filter(fact => fact.similarity > 0.6) // Apenas fatos realmente relevantes
    .map(fact => fact.text)
    .join('\n\n');
}

// === SERVIDOR EXPRESS =================================================
const app = express();
app.use(cors());
app.use(express.json());

app.post('/chat', async (req, res) => {
  const userMsg = (req.body.message || '').slice(0, 2000);

  try {
    // 1. SEMPRE buscamos na base de conhecimento específica
    const relevantFacts = await findRelevantFacts(userMsg);

    // 2. Criamos um PROMPT MESTRE que ensina a IA a priorizar e mesclar
    const finalPrompt = `
      # PERSONA
      Você é o UTFinder, um assistente virtual da UTFPR. Sua personalidade é amigável, prestativa e um pouco descontraída. Use emojis quando apropriado. 😉

      # REGRAS DE RACIOCÍNIO
      1.  **Prioridade Máxima:** Sua primeira fonte de verdade é a seção 'CONTEXTO ESPECÍFICO DA UTFPR'. Baseie sua resposta nela sempre que possível.
      2.  **Complemento com Conhecimento Geral:** Se o contexto específico não for suficiente para responder completamente à pergunta, você **PODE** usar seu conhecimento geral para complementar a resposta.
      3.  **Aviso de Fonte:** Se você usar seu conhecimento geral, você **DEVE** sinalizar isso. Por exemplo: "Na minha base de dados da UTFPR não achei sobre isso, mas de forma geral..." ou "Sobre o prazo, a informação que tenho é X. Já sobre o tempo, como não tenho acesso a dados em tempo real...".
      4.  **Conversa Social:** Para conversas que não são sobre a UTFPR (oi, tudo bem, piadas, etc.), aja naturalmente de acordo com sua persona, sem precisar mencionar o contexto.

      # CONTEXTO ESPECÍFICO DA UTFPR
      ---
      ${relevantFacts || "Nenhum contexto específico encontrado para esta pergunta."}
      ---

      Com base em todas as suas regras, responda a pergunta do usuário.
      Pergunta: "${userMsg}"
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
  console.log(`🚀 Servidor com conhecimento híbrido rodando na porta ${PORT}`);
});