// Aurum Research Engine — https://github.com/aurolabs-ai/aurum
// Licensed under BSL 1.1 — see LICENSE

import OpenAI from 'openai';

const FALLBACK = {
  bottom_line: 'Analysis complete. See full report for details.',
  confidence: { level: 'low', percentage: 30, kent_term: 'Possible' },
};

const SYSTEM_PROMPT = `You are a senior intelligence analyst. Read the report and produce a JSON object with exactly two fields:

1. "bottom_line": The single most important takeaway in 1-2 sentences. Max 40 words. Be direct, no hedging.
2. "confidence": { "level": "high"|"medium"|"low", "percentage": 0-100, "kent_term": "<Sherman Kent scale term>" }

Sherman Kent scale: Almost certain (90-99%), Probable (70-89%), Roughly even (40-69%), Unlikely (11-39%), Remote (1-10%).

Return ONLY valid JSON. No markdown fences, no commentary.`;

/**
 * Lightweight bottom-line generator for Free tier.
 * Returns { bottom_line, confidence: { level, percentage, kent_term } }
 */
export async function generateBottomLine(reportMd, openaiKey, costTracker = {}) {
  const openai = new OpenAI({ apiKey: openaiKey });
  const start = Date.now();

  try {
    const completion = await openai.chat.completions.create({
      model: 'gpt-4.1-mini',
      messages: [
        { role: 'system', content: SYSTEM_PROMPT },
        { role: 'user', content: reportMd.slice(0, 80_000) },
      ],
      temperature: 0.2,
      max_completion_tokens: 500,
      response_format: { type: 'json_object' },
    });

    const usage = completion.usage || {};
    costTracker.model = 'gpt-4.1-mini';
    costTracker.tokens_in = usage.prompt_tokens || 0;
    costTracker.tokens_out = usage.completion_tokens || 0;
    costTracker.cost_usd = (costTracker.tokens_in * 0.40 / 1_000_000) + (costTracker.tokens_out * 1.60 / 1_000_000);
    costTracker.time_ms = Date.now() - start;

    const result = JSON.parse(completion.choices[0].message.content.trim());

    if (!result.bottom_line || !result.confidence) return FALLBACK;
    return result;
  } catch (err) {
    console.error('[BottomLine] Failed:', err.message);
    costTracker.time_ms = Date.now() - start;
    return FALLBACK;
  }
}
