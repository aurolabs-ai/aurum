// Aurum Research Engine — https://github.com/aurolabs-ai/aurum
// Licensed under BSL 1.1 — see LICENSE

import { readdir, readFile } from 'fs/promises';
import { join, dirname } from 'path';
import { fileURLToPath } from 'url';
import OpenAI from 'openai';

const __dirname = dirname(fileURLToPath(import.meta.url));
const TEMPLATES_DIR = join(__dirname, '..', 'templates');
const FALLBACK_TEMPLATE = 'scenario-analysis';

/**
 * Match a research question to the best template using GPT-4.1-mini.
 * Returns template name (without extension).
 */
export async function matchTemplate(question, openaiKey, costsObj = null) {
  // Load available templates
  const files = await readdir(TEMPLATES_DIR);
  const templateFiles = files.filter(f => f.endsWith('.template.md'));

  const templateList = templateFiles.map(f => {
    const name = f.replace('.template.md', '');
    return name;
  });

  if (templateList.length === 0) {
    return FALLBACK_TEMPLATE;
  }

  try {
    const openai = new OpenAI({ apiKey: openaiKey });

    const completion = await openai.chat.completions.create({
      model: 'gpt-4.1-mini',
      temperature: 0,
      max_completion_tokens: 100,
      messages: [
        {
          role: 'system',
          content: `You are a research methodology classifier. Given a research question, select the single best matching research template from the list below.

Available templates:
${templateList.map(t => `- ${t}`).join('\n')}

Template descriptions:
- scenario-analysis: Future scenarios, geopolitical analysis, policy impact, what-if analysis
- market-research: Market sizing, segmentation, trends, TAM/SAM/SOM
- competitor-audit: Competitive landscape analysis, market positioning, company comparisons

Respond with ONLY the template name, nothing else.`,
        },
        {
          role: 'user',
          content: question,
        },
      ],
    });

    if (costsObj) {
      const u = completion.usage || {};
      costsObj.template_match.model = 'gpt-4.1-mini';
      costsObj.template_match.tokens_in = u.prompt_tokens || 0;
      costsObj.template_match.tokens_out = u.completion_tokens || 0;
      costsObj.template_match.cost_usd = (costsObj.template_match.tokens_in * 0.40 / 1_000_000) + (costsObj.template_match.tokens_out * 1.60 / 1_000_000);
    }

    const matched = completion.choices[0].message.content.trim().toLowerCase().replace(/[^a-z0-9-]/g, '');

    if (templateList.includes(matched)) {
      return matched;
    }

    // Fuzzy match: check if response contains a template name
    for (const t of templateList) {
      if (matched.includes(t)) return t;
    }

    return FALLBACK_TEMPLATE;
  } catch (err) {
    return FALLBACK_TEMPLATE;
  }
}

/**
 * Load template content by name.
 */
export async function loadTemplate(templateName) {
  const filePath = join(TEMPLATES_DIR, `${templateName}.template.md`);
  try {
    return await readFile(filePath, 'utf-8');
  } catch {
    return await readFile(join(TEMPLATES_DIR, `${FALLBACK_TEMPLATE}.template.md`), 'utf-8');
  }
}
