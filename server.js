// server.js - Versão FINAL com Roteador de Intenção

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
  console.log('Iniciando construção da base de conhecimento com formato Q&A...');
  const fileContent = fs.readFileSync('./base_conhecimento.txt', 'utf8');
  const qaPairs = fileContent.split('\n\n').filter(p => p.trim());

  if (qaPairs.length === 0) {
    console.warn("Base de conhecimento está vazia ou em formato incorreto.");
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
  console.log(`✅ Base de conhecimento construída com ${knowledgeBase.length} pares de Q&A.`);
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

  return knowledgeBase.slice(0, 2).map(fact => fact.text).join('\n\n');
}

/**
 * ROTEADOR INTELIGENTE
 * Classifica a pergunta do usuário para decidir qual prompt usar.
 */
async function getResponseType(userQuery) {
    const prompt = `
      A pergunta do usuário é específica sobre a Universidade Tecnológica Federal do Paraná (UTFPR), envolvendo cursos, prazos, vestibular, campi, etc., ou é uma conversa geral/social (como "oi", "que dia é hoje?", "qual a cor do céu?")?
      Responda apenas com uma única palavra: "UTFPR" ou "GERAL".

      Pergunta: "${userQuery}"
      Classificação:
    `;
    const result = await chatModel.generateContent(prompt);
    const choice = result.response.text().trim().toUpperCase();
    console.log(`Consulta do usuário classificada como: ${choice}`);
    return choice;
}

// === SERVIDOR EXPRESS =================================================
const app = express();
app.use(cors());
app.use(express.json());

app.post('/chat', async (req, res) => {
  const userMsg = (req.body.message || '').slice(0, 2000);

  try {
    let reply = '';
    let finalPrompt = '';

    // ROTEADOR EM AÇÃO: Decide qual caminho seguir
    const responseType = await getResponseType(userMsg);

    if (responseType === 'UTFPR') {
      // CAMINHO 1: A pergunta é sobre a faculdade, então usamos a busca semântica
      console.log('Roteando para busca semântica (RAG)...');
      const relevantFacts = await findRelevantFacts(userMsg);
      
      finalPrompt = `
        Você é o UTFinder, um assistente prestativo da UTFPR.
        Responda a pergunta do usuário usando APENAS o CONTEXTO abaixo.
        Se o CONTEXTO não for suficiente, diga que não encontrou a informação.

        CONTEXTO:
        ---
        ${relevantFacts}
        ---

        Com base estrita no CONTEXTO acima, responda a pergunta: "${userMsg}"
      `;
    } else {
      // CAMINHO 2: É uma conversa geral, então deixamos a IA responder livremente
      console.log('Roteando para conversa geral...');
      finalPrompt = `
        Você é o UTFinder, um assistente virtual amigável da UTFPR.
        Responda a pergunta do usuário de forma conversacional e natural.
        Se for uma pergunta de conhecimento geral, responda o que souber.
        Não finja ser um humano.

        Pergunta: "${userMsg}"
      `;
    }
    
    const result = await chatModel.generateContent(finalPrompt);
    reply = result.response.text().trim();
    res.json({ reply });

  } catch (err) {
    console.error("Erro no processamento do chat:", err);
    res.status(500).json({ error: err.message || 'Erro ao processar sua requisição.' });
  }
});

// Inicia o servidor
app.listen(PORT, async () => {
  await buildKnowledgeBase();
  console.log(`🚀 Servidor final com roteador inteligente rodando na porta ${PORT}`);
});