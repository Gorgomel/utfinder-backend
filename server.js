// server.js - versão atualizada para Google Gemini API

import dotenv from 'dotenv';
import fs from 'fs';
import express from 'express';
import cors from 'cors';
// Importa a classe do SDK do Google
import { GoogleGenerativeAI } from '@google/generative-ai';

// === CONFIG ======================================================
dotenv.config();

const PORT = process.env.PORT || 3000;
const MODEL_NAME = process.env.GEMINI_MODEL || 'gemini-1.5-pro-latest';
const API_KEY = process.env.GEMINI_API_KEY;
const TXT = fs.readFileSync('./base_conhecimento.txt', 'utf8');

// Inicializa o cliente da API do Google
const genAI = new GoogleGenerativeAI(API_KEY);
const model = genAI.getGenerativeModel({ model: MODEL_NAME });
// ================================================================

const app = express();
app.use(cors());
app.use(express.json());

app.post('/chat', async (req, res) => {
  const userMsg = (req.body.message || '').slice(0, 2000);

  try {
    // Monta o prompt para o Gemini
    const prompt = `Você é o UTFinder. Use APENAS o conteúdo entre '====' como base de conhecimento.
Se o conteúdo citar um link em "[...]", mostre esse link em "Mais informações:" no fim.
====
${TXT}
====
Pergunta do usuário: ${userMsg}`;

    // Gera o conteúdo usando a nova API
    const result = await model.generateContent(prompt);
    const response = await result.response;
    const reply = response.text().trim();

    res.json({ reply });

  } catch (err) {
    console.error("Erro na requisição ao Gemini:", err);
    res.status(500).json({ error: err.message || 'Erro ao consultar a API do Gemini' });
  }
});

app.listen(PORT, () =>
  console.log(`✅ Backend online em http://localhost:${PORT}`)
);