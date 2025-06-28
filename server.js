// server.js - Versão com correção no formato de embedding

import dotenv from 'dotenv';
import fs from 'fs';
import express from 'express';
import cors from 'cors';
import { GoogleGenerativeAI } from '@google/generative-ai';

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
  if (magA === 0 || magB === 0) {
    return 0;
  }
  return dotProduct / (magA * magB);
}

// === CONFIGURAÇÃO ======================================================
dotenv.config();
const PORT = process.env.PORT || 3000;
const API_KEY = process.env.GEMINI_API_KEY;

const genAI = new GoogleGenerativeAI(API_KEY);
const model = genAI.getGenerativeModel({ model: 'gemini-1.5-flash-latest' });
const embeddingModel = genAI.getGenerativeModel({ model: 'text-embedding-004' });

const knowledgeBase = [];

// --- FUNÇÕES DA BASE DE CONHECIMENTO ----------------------------------

async function buildKnowledgeBase() {
  console.log('Iniciando construção da base de conhecimento...');
  const fileContent = fs.readFileSync('./base_conhecimento.txt', 'utf8');
  const facts = fileContent.split('\n').filter(line => line.trim().length > 0);

  if (facts.length === 0) {
    console.warn("Arquivo base_conhecimento.txt está vazio.");
    return;
  }

  // --- CORREÇÃO AQUI ---
  // A API agora espera um objeto com um campo 'content' que, por sua vez,
  // tem um campo 'parts' contendo o texto.
  const requests = facts.map(text => ({
    content: {
      parts: [{ text: text }],
      role: "user" // O papel é obrigatório para o conteúdo
    }
  }));

  const { embeddings } = await embeddingModel.batchEmbedContents({ requests });

  for (let i = 0; i < facts.length; i++) {
    knowledgeBase.push({
      text: facts[i],
      embedding: embeddings[i].values,
    });
  }
  console.log(`✅ Base de conhecimento construída com ${knowledgeBase.length} fatos.`);
}

async function findRelevantFacts(userQuery) {
  if (knowledgeBase.length === 0) return '';
  
  const { embedding } = await embeddingModel.embedContent(userQuery);
  const queryEmbedding = embedding.values;

  for (const fact of knowledgeBase) {
    fact.similarity = cosineSimilarity(queryEmbedding, fact.embedding);
  }

  knowledgeBase.sort((a, b) => b.similarity - a.similarity);

  const topFacts = knowledgeBase
    .slice(0, 3)
    .map(fact => fact.text)
    .join('\n');
    
  console.log('Fatos mais relevantes encontrados:\n', topFacts);
  return topFacts;
}

// === SERVIDOR EXPRESS =================================================
const app = express();
app.use(cors());
app.use(express.json());

app.post('/chat', async (req, res) => {
  const userMsg = (req.body.message || '').slice(0, 2000);

  try {
    const relevantFacts = await findRelevantFacts(userMsg);

    const prompt = `
      Você é o UTFinder, um assistente virtual da UTFPR.
      Responda a pergunta do usuário usando APENAS o CONTEXTO abaixo.
      Se o CONTEXTO não tiver a resposta, diga que não sabe a informação.

      CONTEXTO:
      ---
      ${relevantFacts}
      ---

      Com base estrita no CONTEXTO acima, responda à pergunta do usuário.
      Pergunta: "${userMsg}"
    `;
    
    const result = await model.generateContent(prompt);
    const response = await result.response;
    const reply = response.text().trim();

    res.json({ reply });

  } catch (err) {
    console.error("Erro no processamento do chat:", err);
    res.status(500).json({ error: err.message || 'Erro ao processar sua requisição.' });
  }
});

// Inicia o servidor e constrói a base de conhecimento
app.listen(PORT, async () => {
  await buildKnowledgeBase();
  console.log(`🚀 Servidor rodando na porta ${PORT}`);
});