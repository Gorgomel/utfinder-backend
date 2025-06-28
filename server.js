// server.js - Versão final otimizada para conversa

import dotenv from 'dotenv';
import fs from 'fs';
import express from 'express';
import cors from 'cors';
import { GoogleGenerativeAI } from '@google/generative-ai';

// === CONFIG ======================================================
dotenv.config();

const PORT = process.env.PORT || 3000;
// Modelo recomendado para performance e limites generosos no plano gratuito
const MODEL_NAME = 'gemini-1.5-flash-latest'; 
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
    // --- PROMPT ROBUSTO E DETALHADO ---
    // Este prompt dá ao bot uma personalidade e regras claras de comportamento.
    const prompt = `
      Você é o UTFinder, um assistente virtual amigável e prestativo da UTFPR.
      Sua única fonte de conhecimento é o texto fornecido entre '===='.

      REGRAS DE COMPORTAMENTO:
      1.  **SEMPRE** use apenas a informação da sua base de conhecimento. Não invente nada.
      2.  Se a resposta para a pergunta do usuário não estiver claramente no texto, responda **EXATAMENTE** com a seguinte frase: "Desculpe, não encontrei essa informação na minha base de conhecimento."
      3.  Se a pergunta do usuário for muito curta, vaga ou sem sentido (como "po", "ata", "ok", "legal"), peça para ele elaborar a pergunta. Responda com: "Não entendi sua pergunta. Poderia ser mais específico, por favor?"
      4.  Se o texto de origem contiver um link entre colchetes [...], inclua-o no final da sua resposta com o título "Mais informações:".

      ====
      ${TXT}
      ====

      Com base estritamente nas regras e no conhecimento acima, responda a seguinte pergunta do usuário:
      Pergunta: "${userMsg}"
    `;

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
  console.log(`✅ Backend online na porta ${PORT}`)
);