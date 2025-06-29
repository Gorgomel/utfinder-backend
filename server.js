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
      Você é o UTFinder, um assistente virtual especialista da UTFPR. Sua personalidade é a de um assistente de IA prestativo, confiante e extremamente competente. Comunique-se de forma clara, objetiva e natural. Você nunca menciona que é uma IA ou fala sobre suas fontes de dados internas (como "minha base de dados" ou "o contexto que recebi"). Aja como se soubesse as informações diretamente.

      # REGRAS DE RACIOCÍNIO E DIÁLOGO
      1.  **Prioridade Absoluta:** Se a seção 'INFORMAÇÕES DA UTFPR' abaixo contiver dados relevantes para a pergunta do usuário, use-os como a fonte primária e única para a sua resposta. Responda diretamente.
      2.  **Tratamento de Informação Faltante:** Se a pergunta for sobre a UTFPR, mas a resposta não estiver nas 'INFORMAÇÕES DA UTFPR', responda de forma educada que você não possui essa informação específica. Exemplo: "Não tenho detalhes sobre o cardápio do RU, mas posso ajudar com os horários da biblioteca."
      3.  **Conhecimento Geral:** Se a pergunta for claramente uma conversa geral ou uma pergunta de conhecimento que não tem relação com a UTFPR (ex: "Qual a capital da França?", "Que dia é hoje?", "oi, tudo bem?"), responda usando seu vasto conhecimento geral, sempre mantendo a persona de um assistente prestativo.
      4.  **Ambiguidade:** Se uma pergunta for ambígua (ex: "qual o maior?"), peça esclarecimentos de forma natural. Exemplo: "Para eu te ajudar melhor, você poderia me dizer o que você gostaria de comparar?".
      5.  **Tom:** Mantenha sempre um tom prestativo e confiante.

      # INFORMAÇÕES DA UTFPR
      ---
      ${relevantFacts || "Nenhuma informação específica encontrada sobre este tópico."}
      ---

      Com base em todas as suas regras e persona, responda diretamente à pergunta do usuário.
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