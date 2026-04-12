// Aurum Research Engine — https://github.com/aurolabs-ai/aurum
// Licensed under BSL 1.1 — see LICENSE

import OpenAI from 'openai';

const REDUCE_MODEL = 'gpt-4.1-mini';

const PRICING = {
  'gpt-4.1-mini': { in: 0.4, out: 1.6 },
};

/**
 * Reduce phase: takes raw scratchpad and produces a STRUCTURED, DEDUPLICATED
 * evidence pack organized by section.
 *
 * For each section in the plan:
 *   1. Extract relevant claims/data from scratchpad
 *   2. Deduplicate
 *   3. Score sources (primary institutional > news > opinion)
 *   4. Mark contradictions
 *   5. Rank by relevance
 *
 * Returns: Map of section_name -> { claims, sources, contradictions, key_stats }
 */
export async function reduceData(scratchpad, sections, openaiKey, costTracker = {}) {
  const startTime = Date.now();
  console.log(`[Reduce] === REDUCE PHASE: ${sections.length} sections ===`);
  console.log(`[Reduce] Scratchpad: ${scratchpad.split(/\s+/).length} words`);

  let totalTokensIn = 0;
  let totalTokensOut = 0;
  let totalCost = 0;

  const client = new OpenAI({ apiKey: openaiKey });

  // Process sections in parallel batches of 3
  const BATCH_SIZE = 3;
  const evidencePack = {};

  for (let i = 0; i < sections.length; i += BATCH_SIZE) {
    const batch = sections.slice(i, i + BATCH_SIZE);
    const batchNum = Math.floor(i / BATCH_SIZE) + 1;
    console.log(`[Reduce] Batch ${batchNum}: processing ${batch.length} sections...`);

    const results = await Promise.allSettled(
      batch.map(sectionName => extractForSection(client, sectionName, scratchpad, sections))
    );

    for (let j = 0; j < batch.length; j++) {
      const sectionName = batch[j];
      const result = results[j];

      if (result.status === 'fulfilled' && result.value) {
        const { parsed, usage } = result.value;
        evidencePack[sectionName] = parsed;

        if (usage) {
          const tokIn = usage.prompt_tokens || 0;
          const tokOut = usage.completion_tokens || 0;
          const p = PRICING[REDUCE_MODEL];
          const cost = (tokIn * p.in / 1_000_000) + (tokOut * p.out / 1_000_000);
          totalTokensIn += tokIn;
          totalTokensOut += tokOut;
          totalCost += cost;
        }

        const claimCount = parsed.claims?.length || 0;
        const statCount = parsed.key_stats?.length || 0;
        const contraCount = parsed.contradictions?.length || 0;
        console.log(`[Reduce]   "${sectionName}": ${claimCount} claims, ${statCount} stats, ${contraCount} contradictions`);
      } else {
        console.log(`[Reduce]   "${sectionName}": FAILED - ${result.reason?.message || 'unknown error'}`);
        evidencePack[sectionName] = { claims: [], sources: [], contradictions: [], key_stats: [] };
      }
    }
  }

  const elapsedMs = Date.now() - startTime;
  console.log(`[Reduce] Complete: ${Object.keys(evidencePack).length} sections reduced in ${(elapsedMs / 1000).toFixed(1)}s, $${totalCost.toFixed(4)}`);

  // Update cost tracker
  costTracker.model = REDUCE_MODEL;
  costTracker.tokens_in = totalTokensIn;
  costTracker.tokens_out = totalTokensOut;
  costTracker.cost_usd = totalCost;
  costTracker.time_ms = elapsedMs;

  return evidencePack;
}

async function extractForSection(client, sectionName, scratchpad, allSections) {
  // Truncate scratchpad to fit context — 150k chars max per call
  const maxChars = 150_000;
  const trimmed = scratchpad.length > maxChars
    ? scratchpad.slice(0, maxChars) + '\n[... truncated ...]'
    : scratchpad;

  const response = await client.chat.completions.create({
    model: REDUCE_MODEL,
    temperature: 0,
    max_completion_tokens: 4000,
    response_format: { type: 'json_object' },
    messages: [
      {
        role: 'system',
        content: `You are a research data reducer. Given a raw research scratchpad and a specific report section name, extract ONLY the data relevant to that section.

Your job is classification and extraction, not synthesis. Be thorough and precise.

SOURCE TIER RANKING (assign to each claim):
- "tier1_institutional": Government agencies, central banks, international orgs (IMF, World Bank, UN), academic journals
- "tier2_industry": Industry reports (McKinsey, Gartner, Statista), major financial institutions, company filings
- "tier3_news": Reuters, Bloomberg, AP, major newspapers
- "tier4_opinion": Blogs, opinion pieces, analyst commentary, social media

OUTPUT FORMAT (valid JSON):
{
  "claims": [
    {
      "text": "Specific factual claim with numbers/dates",
      "source_name": "Source Name",
      "source_url": "https://...",
      "source_tier": "tier1_institutional",
      "year": 2025,
      "relevance": 1-10
    }
  ],
  "key_stats": [
    {
      "stat": "$X billion market size by 2027",
      "source_name": "Source Name",
      "source_url": "https://...",
      "year": 2025
    }
  ],
  "contradictions": [
    {
      "claim_a": "Source A says X",
      "claim_b": "Source B says Y",
      "source_a_url": "https://...",
      "source_b_url": "https://..."
    }
  ],
  "sources": [
    {
      "name": "Source Name",
      "url": "https://...",
      "tier": "tier1_institutional",
      "claim_count": 3
    }
  ]
}

RULES:
- Extract EVERY relevant data point — do not summarize or compress
- Deduplicate: if the same fact appears from multiple searches, keep the best-sourced version
- Include the source URL for EVERY claim — no URL = drop the claim
- Sort claims by relevance (10 = critical for this section, 1 = tangentially related)
- Flag contradictions explicitly
- If no data is relevant to this section, return empty arrays`,
      },
      {
        role: 'user',
        content: `SECTION TO EXTRACT FOR: "${sectionName}"

ALL REPORT SECTIONS (for context on scope): ${allSections.join(', ')}

RAW RESEARCH SCRATCHPAD:
${trimmed}`,
      },
    ],
  });

  const raw = response.choices[0]?.message?.content || '{}';
  let parsed;
  try {
    parsed = JSON.parse(raw);
  } catch (parseErr) {
    console.log(`[Reduce] JSON parse failed for "${sectionName}": ${parseErr.message}, attempting repair`);
    // Try to repair truncated JSON by finding last valid closing brace
    parsed = null;
    const lastBrace = raw.lastIndexOf('}');
    if (lastBrace > 0) {
      try {
        parsed = JSON.parse(raw.slice(0, lastBrace + 1));
        console.log(`[Reduce] JSON repair succeeded for "${sectionName}" (truncated at char ${lastBrace})`);
      } catch {
        console.log(`[Reduce] JSON repair failed for "${sectionName}", trying smaller chunk`);
      }
    }
    // If repair failed, try to extract partial arrays from the raw text
    if (!parsed) {
      parsed = {};
      try {
        const claimsMatch = raw.match(/"claims"\s*:\s*\[[\s\S]*?\]/);
        if (claimsMatch) parsed.claims = JSON.parse(`{${claimsMatch[0]}}`).claims;
      } catch {}
      try {
        const statsMatch = raw.match(/"key_stats"\s*:\s*\[[\s\S]*?\]/);
        if (statsMatch) parsed.key_stats = JSON.parse(`{${statsMatch[0]}}`).key_stats;
      } catch {}
      const partialCount = (parsed.claims?.length || 0) + (parsed.key_stats?.length || 0);
      if (partialCount > 0) {
        console.log(`[Reduce] Extracted ${partialCount} partial items from malformed JSON for "${sectionName}"`);
      } else {
        console.log(`[Reduce] No salvageable data from "${sectionName}", returning empty evidence`);
      }
    }
  }
  return {
    parsed: {
      claims: parsed.claims || [],
      sources: parsed.sources || [],
      contradictions: parsed.contradictions || [],
      key_stats: parsed.key_stats || [],
    },
    usage: response.usage,
  };
}
