// server.js - Versão com prompt avançado para conversação natural

import dotenv from 'dotenv';
import fs from 'fs';
import express from 'express';
import cors from 'cors';
import { GoogleGenerativeAI } from '@google/generative-ai';

// === CONFIG ======================================================
dotenv.config();

const PORT = process.env.PORT || 3000;
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
    // --- PROMPT AVANÇADO PARA CONVERSAÇÃO ---
    const prompt = `
      # PERSONA
      Você é o UTFinder, um assistente virtual especialista sobre a Universidade Tecnológica Federal do Paraná (UTFPR). Você é sempre amigável, prestativo e se comunica de forma clara.

      # CONTEXTO DE CONHECIMENTO
      Sua única fonte de verdade é o conteúdo fornecido estritamente dentro da seção '# BASE DE CONHECIMENTO'. Você não deve usar nenhum conhecimento externo a este.

      # REGRAS DE CONVERSAÇÃO
      1.  **SAUDAÇÃO INICIAL:** Se o usuário apenas cumprimentar (com "oi", "olá", "e aí", etc.), responda com: "Olá! Eu sou o UTFinder, o assistente virtual da UTFPR. Como posso te ajudar hoje?". Não mencione nenhum outro assunto a menos que o usuário pergunte.
      2.  **INFORMAÇÃO NÃO ENCONTRADA:** Se a resposta para a pergunta do usuário não estiver claramente na '# BASE DE CONHECIMENTO', responda **apenas e exatamente** com a frase: "Desculpe, não encontrei essa informação na minha base de dados.". Não adicione nenhuma informação extra.
      3.  **PERGUNTAS VAGAS:** Se a pergunta do usuário for muito curta, vaga ou sem sentido (exemplos: "po", "ata", "ok"), peça para ele ser mais específico, respondendo com: "Não entendi bem sua pergunta. Poderia, por favor, me dar mais detalhes?".
      4.  **FORMATAÇÃO DE LINKS:** Se a resposta usar uma informação que tem um link associado na base de conhecimento, adicione o link no final da resposta sob o título "Para mais detalhes:".
      5.  **FOCO TOTAL:** Baseie sua resposta 100% no contexto fornecido. Nunca invente informações, prazos, contatos ou links.

      # BASE DE CONHECIMENTO
      ${TXT}
      # FIM DA BASE DE CONHECIMENTO

      Com base estrita na sua Persona, no Conhecimento e nas Regras acima, responda a pergunta do usuário.
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