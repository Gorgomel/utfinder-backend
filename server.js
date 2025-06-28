// server.js - versão ESModule para OpenAI v4+

import dotenv from 'dotenv';
import fs from 'fs';
import express from 'express';
import cors from 'cors';
import OpenAI from 'openai';

// === CONFIG ======================================================
dotenv.config();

const PORT  = process.env.PORT || 3000;
const MODEL = process.env.OPENAI_MODEL || 'gpt-3.5-turbo';
const TXT   = fs.readFileSync('./base_conhecimento.txt', 'utf8');

const openai = new OpenAI({
  apiKey: process.env.OPENAI_API_KEY
});
// ================================================================

const app = express();
app.use(cors());
app.use(express.json());

app.post('/chat', async (req, res) => {
  const userMsg = (req.body.message || '').slice(0, 2000);

  try {
    const completion = await openai.chat.completions.create({
      model: MODEL,
      temperature: 0.2,
      messages: [
        {
          role: 'system',
          content:
`Você é o UTFinder. Use APENAS o conteúdo entre '====' como base de conhecimento.
Se o conteúdo citar um link em "[...]", mostre esse link em "Mais informações:" no fim.
====
${TXT}
====`
        },
        { role: 'user', content: userMsg }
      ]
    });

    const reply = completion.choices[0].message.content.trim();
    res.json({ reply });
  } catch (err) {
    console.error(err?.error || err);
    res.status(500).json({ error: 'Erro ao consultar ChatGPT' });
  }
});

app.listen(PORT, () =>
  console.log(`✅ Backend online em http://localhost:${PORT}`)
);
