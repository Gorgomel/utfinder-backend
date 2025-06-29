// server.js - Versão Definitiva com Síntese de Informações

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
  console.log('Iniciando construção da base de conhecimento com formato Q&A...');
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
    knowledgeBase.push({ text: qaPairs[i], embedding: embeddings[i].values });
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

  // Aumentamos para 5 para capturar mais contexto para síntese
  return knowledgeBase
    .slice(0, 5)
    .filter(fact => fact.similarity > 0.65) // Limiar um pouco mais alto
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
    const relevantFacts = await findRelevantFacts(userMsg);

    // O PROMPT FINAL, COM INSTRUÇÃO DE SÍNTESE
    const finalPrompt = `
      # PERSONA
      Você é o UTFinder, um assistente especialista da UTFPR. Sua comunicação é clara, prestativa e confiante. Você nunca menciona sua base de dados ou que é uma IA.

      # REGRAS DE RACIOCÍNIO
      1.  **SÍNTESE DE INFORMAÇÃO:** Sua principal tarefa é responder a pergunta do usuário. Se o CONTEXTO abaixo contiver múltiplos fatos relevantes (ex: cursos em diferentes campi), **sintetize-os em uma única resposta completa e bem organizada**. Não liste os fatos separadamente.
      2.  **PRECISÃO:** Baseie sua resposta estritamente no CONTEXTO. Não adicione informações que não estejam lá.
      3.  **INFORMAÇÃO FALTANTE:** Se o CONTEXTO não contiver absolutamente nenhuma informação relevante para responder à pergunta, diga de forma educada que não possui essa informação específica. Ex: "Não encontrei informações sobre X."
      4.  **CONVERSA GERAL:** Se a pergunta for um bate-papo casual (oi, tudo bem, etc.), responda de forma natural e amigável.

      # CONTEXTO
      ---
      ${relevantFacts || "Nenhum contexto relevante encontrado."}
      ---

      Com base em todas as suas regras, e priorizando a SÍNTESE, responda à pergunta do usuário.
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
  console.log(`🚀 Servidor com capacidade de síntese rodando na porta ${PORT}`);
});