// server.js - Versão final com processamento em lotes (chunking)

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
  
  // --- CORREÇÃO AQUI: Processamento em Lotes (Chunking) ---
  const batchSize = 100;
  console.log(`Total de fatos a processar: ${facts.length}. Processando em lotes de ${batchSize}...`);

  for (let i = 0; i < facts.length; i += batchSize) {
    const batchFacts = facts.slice(i, i + batchSize);
    console.log(`Processando lote de ${i} a ${i + batchFacts.length - 1}...`);

    const contents = batchFacts.map(text => ({ parts: [{ text }], role: "user" }));
    const { embeddings } = await embeddingModel.batchEmbedContents({
        requests: contents.map(content => ({ content, taskType: TaskType.RETRIEVAL_DOCUMENT }))
    });

    for (let j = 0; j < batchFacts.length; j++) {
      knowledgeBase.push({ text: batchFacts[j], embedding: embeddings[j].values });
    }
  }
  // --- FIM DA CORREÇÃO ---
  
  console.log(`✅ Base de conhecimento construída com ${knowledgeBase.length} fatos.`);
}

async function findRelevantFactsHyDE(userQuery) {
    if (knowledgeBase.length === 0) return '';
  
    const promptHyDE = `Escreva um pequeno parágrafo que responda a seguinte pergunta, mesmo que você não saiba a resposta exata: "${userQuery}"`;
    const hypotheticalAnswerResult = await chatModel.generateContent(promptHyDE);
    const hypotheticalAnswer = hypotheticalAnswerResult.response.text();
    
    const { embedding } = await embeddingModel.embedContent(hypotheticalAnswer);
    const queryEmbedding = embedding.values;
  
    for (const fact of knowledgeBase) {
      fact.similarity = cosineSimilarity(queryEmbedding, fact.embedding);
    }
  
    knowledgeBase.sort((a, b) => b.similarity - a.similarity);
  
    const topFacts = knowledgeBase
      .slice(0, 3)
      .filter(fact => fact.similarity > 0.7)
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
      Você é o UTFinder, um assistente virtual especialista da Universidade Tecnológica Federal do Paraná (UTFPR). Sua personalidade é prestativa, confiante e sua comunicação é clara e natural.

      # REGRAS DE DIÁLOGO E RACIOCÍNIO
      1.  **PROIBIÇÃO DE META-COMENTÁRIO:** Você NUNCA deve mencionar sua base de dados, o contexto que recebeu, ou que você é uma IA. Aja como se soubesse a informação diretamente. Frases como "na minha base de dados", "no contexto que recebi" ou "não tenho informações específicas sobre a UTFPR" são estritamente proibidas.
      2.  **USO DO CONTEXTO:** Se a seção 'INFORMAÇÕES DA UTFPR' abaixo contiver dados relevantes para a pergunta, sua resposta DEVE ser baseada estritamente neles.
      3.  **APRESENTAÇÃO DE LINKS:** Se um fato no CONTEXTO contiver um link entre colchetes (ex: [https://...]), você DEVE incluí-lo no final da sua resposta, sob o título "Fonte:" ou "Para mais detalhes:".
      4.  **INFORMAÇÃO INEXISTENTE:** Se a pergunta for sobre a UTFPR mas o CONTEXTO não ajudar, responda de forma educada que você não possui essa informação específica. Exemplo: "Não tenho detalhes sobre o cardápio do RU desta semana."
      5.  **CONVERSA GERAL:** Se a pergunta for um bate-papo casual (oi, como vai, que dia é hoje?), responda naturalmente usando seu conhecimento geral, sem mencionar a UTFPR, a menos que o usuário o faça.

      # INFORMAÇÕES DA UTFPR (CONTEXTO)
      ---
      ${relevantFacts || "Nenhum contexto relevante para esta pergunta."}
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
  console.log(`🚀 Servidor HyDE rodando na porta ${PORT}`);
});