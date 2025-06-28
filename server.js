// server.js - Versão com Busca Semântica e Embeddings

import dotenv from 'dotenv';
import fs from 'fs';
import express from 'express';
import cors from 'cors';
import { GoogleGenerativeAI } from '@google/generative-ai';
import { similarity } from 'simple-linalg';

// === CONFIGURAÇÃO ======================================================
dotenv.config();
const PORT = process.env.PORT || 3000;
const API_KEY = process.env.GEMINI_API_KEY;

// Inicializa os clientes da API do Google
const genAI = new GoogleGenerativeAI(API_KEY);
const model = genAI.getGenerativeModel({ model: 'gemini-1.5-flash-latest' });
const embeddingModel = genAI.getGenerativeModel({ model: 'text-embedding-004' });

// Armazenamento em memória para nossa base de conhecimento "vetorizada"
const knowledgeBase = [];

// --- FUNÇÕES DA BASE DE CONHECIMENTO ----------------------------------

/**
 * Processa o arquivo .txt e cria os embeddings para cada linha.
 * Isso é feito apenas uma vez, quando o servidor inicia.
 */
async function buildKnowledgeBase() {
  console.log('Iniciando construção da base de conhecimento...');
  const fileContent = fs.readFileSync('./base_conhecimento.txt', 'utf8');
  const facts = fileContent.split('\n').filter(line => line.trim().length > 0);

  // Pega todos os fatos e transforma em embeddings
  const { embeddings } = await embeddingModel.batchEmbedContents({
    requests: facts.map(text => ({ content: text })),
  });

  // Salva o texto original junto com seu vetor de embedding
  for (let i = 0; i < facts.length; i++) {
    knowledgeBase.push({
      text: facts[i],
      embedding: embeddings[i].values,
    });
  }
  console.log(`✅ Base de conhecimento construída com ${knowledgeBase.length} fatos.`);
}

/**
 * Encontra os fatos mais relevantes na base de conhecimento para uma dada pergunta.
 * @param {string} userQuery A pergunta do usuário.
 * @returns {string} Uma string contendo os fatos mais relevantes.
 */
async function findRelevantFacts(userQuery) {
  if (knowledgeBase.length === 0) return '';

  // 1. Cria o embedding para a pergunta do usuário
  const { embedding } = await embeddingModel.embedContent(userQuery);
  const queryEmbedding = embedding.values;

  // 2. Calcula a similaridade entre a pergunta e cada fato na base
  for (const fact of knowledgeBase) {
    fact.similarity = similarity(queryEmbedding, fact.embedding);
  }

  // 3. Ordena os fatos pela similaridade (do maior para o menor)
  knowledgeBase.sort((a, b) => b.similarity - a.similarity);

  // 4. Pega os 3 fatos mais relevantes e formata para o prompt
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
    // Passo 1: Encontrar os fatos mais relevantes usando a busca semântica
    const relevantFacts = await findRelevantFacts(userMsg);

    // Passo 2: Construir um prompt focado apenas com a informação relevante
    const prompt = `
      Você é o UTFinder, um assistente virtual amigável da UTFPR.
      Sua tarefa é responder a pergunta do usuário usando APENAS o CONTEXTO fornecido abaixo.
      Se o CONTEXTO não contiver a resposta, diga que não encontrou a informação.
      Não use nenhum conhecimento prévio.

      CONTEXTO:
      ---
      ${relevantFacts}
      ---

      Com base estrita no CONTEXTO acima, responda a pergunta do usuário.
      Pergunta: "${userMsg}"
    `;
    
    // Passo 3: Gerar a resposta final com o modelo de chat
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
  console.log(`🚀 Servidor rodando em http://localhost:${PORT}`);
});