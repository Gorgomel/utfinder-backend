// server.js - Versão Final com Multi-Query HyDE para máxima precisão

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
 * Usa a técnica Multi-Query HyDE para encontrar os fatos mais relevantes.
 */
async function findRelevantFactsMultiQuery(userQuery) {
    if (knowledgeBase.length === 0) return '';
  
    // 1. Gera 3 variações da pergunta do usuário para uma busca mais ampla
    const multiQueryPrompt = `Gere 3 variações da seguinte pergunta de usuário, mantendo o mesmo significado. Separe cada variação com '|||'.
    Pergunta original: "${userQuery}"
    Variações:`;
    const multiQueryResult = await chatModel.generateContent(multiQueryPrompt);
    const queries = [userQuery, ...multiQueryResult.response.text().split('|||').map(q => q.trim())];
  
    // 2. Cria embeddings para todas as variações da pergunta
    const { embeddings } = await embeddingModel.batchEmbedContents({
      requests: queries.map(q => ({
        content: { parts: [{ text: q }], role: "user" },
        taskType: TaskType.RETRIEVAL_QUERY,
      })),
    });
    const queryEmbeddings = embeddings.map(e => e.values);
  
    // 3. Para cada fato na base, encontra a sua MELHOR similaridade contra TODAS as variações da pergunta
    for (const fact of knowledgeBase) {
      let maxSimilarity = 0;
      for (const queryEmbedding of queryEmbeddings) {
        const currentSimilarity = cosineSimilarity(queryEmbedding, fact.embedding);
        if (currentSimilarity > maxSimilarity) {
          maxSimilarity = currentSimilarity;
        }
      }
      fact.similarity = maxSimilarity;
    }
  
    knowledgeBase.sort((a, b) => b.similarity - a.similarity);
  
    const topFacts = knowledgeBase
      .slice(0, 4) // Pega os 4 melhores fatos
      .filter(fact => fact.similarity > 0.7)
      .map(fact => fact.text)
      .join('\n\n');
  
    console.log('--- Multi-Query ---');
    console.log('Variações de Busca Geradas:', queries.join(' | '));
    console.log('Fatos Relevantes Encontrados:', topFacts || 'Nenhum');
    console.log('-------------------');
  
    return topFacts;
}

// === SERVIDOR EXPRESS =================================================
const app = express();
app.use(cors());
app.use(express.json());

app.post('/chat', async (req, res) => {
  const userMsg = (req.body.message || '').slice(0, 2000);

  try {
    const relevantFacts = await findRelevantFactsMultiQuery(userMsg);

    const finalPrompt = `
      # PERSONA
      Você é o UTFinder, um assistente especialista da UTFPR. Sua comunicação é clara, direta e sempre prestativa.

      # INSTRUÇÕES
      - Sua principal tarefa é responder a pergunta do usuário com base no CONTEXTO.
      - Se o CONTEXTO contiver múltiplos fatos relevantes, sintetize-os em uma resposta única e coesa.
      - Se o CONTEXTO não for relevante, responda usando seu conhecimento geral de forma natural e amigável.
      
      # CONTEXTO
      ---
      ${relevantFacts || "Nenhum."}
      ---

      # PERGUNTA
      "${userMsg}"
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
  console.log(`🚀 Servidor Multi-Query HyDE rodando na porta ${PORT}`);
});