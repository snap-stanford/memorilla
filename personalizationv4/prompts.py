"""Prompt templates for the four model calls of the PersonalizationV4 generation pipeline."""

CATEGORIES_TEMPLATE = """You are a data analysis specialist. Your task is to analyze a detailed user persona and identify the {num_categories} most important dimensions or categories that should be tested in an evaluation set.

**User Persona:**
{persona}

**Instructions:**
1. Read the entire persona carefully and identify its key dimensions
2. Generate exactly {num_categories} category descriptions
3. Each category should:
   - Be specific to this persona (not generic)
   - Cover a distinct aspect of their life, personality, or behavior
   - Be testable through multiple-choice questions
   - Be written as a brief descriptive phrase (3-8 words)

**Category Types to Consider:**
- Professional life and work patterns
- Specific hobbies or interests mentioned
- Communication and personality traits
- Relationships and social patterns
- Food, music, media, or other taste preferences
- Hidden patterns, contradictions, or secret projects
- Temporal evolution (how they've changed over time)
- Technology interaction patterns
- Daily routines and habits
- Location-specific knowledge
- Specific skills or expertise areas

**Output Format:**
Return a numbered list of exactly {num_categories} category descriptions, one per line.

Example output format:
1. Work Ethics & Procrastination Patterns
2. Music Preferences for Different Contexts
3. Dietary Habits & Food Contradictions
4. Secret Graphic Novel Project
5. Evolution of AI Trust Over Time

**Task:**
Analyze the persona and generate {num_categories} specific, testable question categories."""

TOPICS_TEMPLATE = """You are a data generation specialist. Your task is to create a list of {num_topics} unique, single-sentence summaries for potential chat interactions with the user described in the persona below.
Your goal is to ensure maximum diversity in the generated summaries, covering all facets of the user's life and personality as described in their profile.

**User Persona:**
{persona}

**Instructions:**
1. **Read and internalize the entire persona document.** Your summaries MUST reflect its details.
2. **Generate a list of exactly {num_topics} unique sentences.** Each sentence should describe a plausible interaction.
3. **Ensure comprehensive coverage of the persona:**
   * **Work Life:** Include tasks related to their professional work, client interactions, brainstorming, and any mentioned work habits.
   * **Hobbies & Tastes:** Create scenarios covering their interests, music tastes, dietary habits, fitness activities, and specific media preferences.
   * **Personality & Communication:** Generate summaries that reflect their different moods, humor style, and communication patterns.
   * **Temporal Evolution:** Crucially, create a mix of interactions from their **early, skeptical phase** with the AI and their **later, reliant phase**.
   * **Secret Projects:** Weave in scenarios where they make seemingly random research requests that are secretly related to hidden projects. Do NOT mention the secret projects directly in the summaries.
4. **Format:** Output the list as a simple text file with each sentence on a new line, numbered.

**Example Format:**
1. User asks for a high-energy, lyric-free playlist for a long work session.
2. User complains sarcastically about a frustrating request.
3. User asks the assistant to find three quick recipes matching their dietary preferences.
4. In a series of rapid messages, user brainstorms ideas for a project.
5. User asks for a list of films by a specific director and where to stream them.
6. User skeptically asks the assistant to confirm information they already know.
7. User asks the assistant to draft an important message.
8. (Secret Project) User asks for reference materials on a seemingly random topic.

**Task:**
Follow the instructions above to generate a list of {num_topics} unique, single-sentence summaries for potential chat interactions."""

CHAT_TEMPLATE = """You are a dialogue generation specialist. Your task is to generate a single, two-turn chat interaction between "Leo" and "Assistant" based on a provided scenario and a detailed persona profile.

**Context:**
1. **Persona Profile:**
{persona}

2. **Scenario:**
{topic}

**Instructions:**
1. **Embody the Persona:** Generate Leo's message precisely according to his personality and communication style as described in the profile (tone, humor, emoji use, etc.). Remember his directness when focused on work.
2. **Embody the Assistant:** The Assistant's response should be short, helpful, professional, and slightly neutral. It should not try to mimic Leo's sarcasm. It should be efficient and accurate.
3. **Maintain Implicitness:** Do not explicitly state Leo's traits. Show, don't tell. For example, if the scenario is about food, don't say "Here are vegetarian options for you." Instead, just provide the options as requested.
4. **Strict Output Format:** The output must be exactly two lines, formatted as follows:
Leo: [Generated message for Leo]
Assistant: [Generated message for Assistant]

**Example Execution:**
**INPUT SCENARIO:** "Leo is looking for a dinner spot for Friday night to celebrate finishing a big project."
**EXPECTED OUTPUT:**
Leo: The big client project is finally off my plate. I feel like I deserve something that isn't a kale salad. Thinking a really, really good burger. What's the best spot in my area for that? And I mean proper artisanal, not some fast-food chain.\\n
Assistant: Congratulations on the project completion, Leo. For a high-quality, artisanal burger near Queen West, a top-rated option is The Burger's Priest on Queen St W. Another excellent choice, known for its craftsmanship and a short streetcar ride away, is Matty's Patty's Burger Club.

**Task:**
Generate the two-turn interaction."""

QUESTIONS_TEMPLATE = """You are a data generation specialist and psychometrician creating a High-Difficulty evaluation set for an advanced AI memory system. Your goal is to generate exactly {num_questions} high-quality, multiple-choice Question-Answer pairs based on the provided Persona Profile.

**Context:**
1. **Target Category:** All questions generated in this batch MUST strictly relate to the following category:
**{category}**

2. **Persona Profile (Ground Truth):**
{persona}

**Instructions:**
1. **Source of Truth:** The Persona Profile is your absolute source of truth. Each question must require synthesizing information or understanding nuance from the persona.
2. **Competitive Distractors:** Every choice (A-E) must be a reasonable, professional, or logical preference for a human. Never use obviously wrong, absurd, or incompetent distractors.
3. **Style Parity:** Ensure all choices are roughly the same length and level of detail. Do not use specific proper nouns or extra nuance in the correct answer unless they are also present in the distractors.
4. **Randomize Correct Position:** Distribute correct answers evenly across A, B, C, D, and E throughout the batch.

**Strategies for High Difficulty (MUST FOLLOW):**
1. **Test Preference, Not Competence:** Instead of testing Good vs Bad, test Option A vs Option B where both are valid philosophies. If Leo prefers 1:1 meetings, distractors should be Small group workshops or Async video updates, not Ignoring the client.
2. **Avoid Semantic Opposites:** If the persona hates jargon, do not make the distractor Use lots of jargon. Instead, use Use highly technical academic language or Use data-heavy metrics.
3. **The Best-Practice Trap:** Ensure distractors include industry-standard best practices that most people in the persona's role would choose, but which this specific persona rejects.
4. **Contextual Displacement:** Use information that is true about the persona but applies it to the wrong situation or frequency.
5. **Homogeneous Detail:** If the correct answer includes a specific Why or How, the distractors must also include equally plausible Whys and Hows.

**Anti-Guessing Requirements (MUST FOLLOW):**
Research shows that language models can guess MCQ answers by reading only the choices without seeing the question or source material. They detect statistical patterns such as length differences, specificity imbalances, and outlier phrasing. You must eliminate these patterns.
1. **Length Matching:** All five choices must have the same word count within a tolerance of two words. Before finalizing, count the words in each choice and adjust until they match.
2. **Structural Mirroring:** All choices must follow the same grammatical template. If the correct answer follows the pattern A private space with [modifier] [noun] paired with [adjective] [activity], then all distractors must follow this identical structure.
3. **Specificity Parity:** If the correct answer contains a proper noun, all distractors must contain proper nouns. If the correct answer contains a number or statistic, all distractors must contain numbers or statistics. If the correct answer contains a because clause, all distractors must contain because clauses.
4. **No Outliers:** Read all five choices together as a group. If any single choice stands out as more detailed, more nuanced, more hedged, more confident, or more professional-sounding than the others, rewrite until all choices are indistinguishable in tone and style.
5. **Prevent Semantic Clustering:** Ensure distractors do not cluster around one theme while the correct answer represents a different theme. Spread distractors across different but equally plausible approaches.
6. **Require Synthesis:** Questions must require combining two or more facts from the persona profile. Avoid questions that can be answered by recalling a single fact.
   - Weak: What music does Leo prefer?
   - Strong: When Leo faces a tight deadline on a branding project requiring deep creative work, which environmental setup would he choose?
7. **Scenario Embedding:** Embed each question in a specific, realistic scenario that requires applying persona knowledge to a novel situation rather than simple recall.
8. **Neutral Question Stems:** The question stem must not hint at characteristics of the answer. Avoid framing like Given Leo's introverted nature and need for high-energy focus, which leaks information about the correct choice.

**Self-Validation:**
Before outputting each question, verify the following:
- All five choices have word counts within two words of each other
- All five choices follow the same grammatical structure
- All five choices have the same level of concrete detail
- No single choice stands out when reading only the choices without the question
- All distractors represent approaches a reasonable professional might authentically prefer
- The question requires combining multiple persona facts to answer correctly

**Output Format:**

<QUESTION>
[Scenario-embedded question requiring synthesis of multiple persona attributes]
<CHOICE_A>
[Option following the shared structure template]
<CHOICE_B>
[Option following the shared structure template]
<CHOICE_C>
[Option following the shared structure template]
<CHOICE_D>
[Option following the shared structure template]
<CHOICE_E>
[Option following the shared structure template]
<CORRECT_CHOICE>
[Letter]
<RATIONALE>
[Why the correct answer is correct based on the persona, and why each distractor represents a plausible but incorrect choice for this specific persona]
<WORD_COUNTS>
A: [n] | B: [n] | C: [n] | D: [n] | E: [n]

**Example of Weak vs Strong Construction:**

Weak (Guessable):
Question: How does Leo prefer to receive feedback on his design work?
Choice A: Through detailed written reports with specific actionable items and deadlines.
Choice B: Quick verbal feedback.
Choice C: In scheduled one-on-one sessions where he can discuss and ask clarifying questions in depth.
Choice D: Via async video messages.
Choice E: Public team critiques.

Problems: Choice C is the longest and most detailed. Choices B and E are significantly shorter. Word counts vary from 3 to 18. Choice C sounds more thoughtful than the others.

Strong (Hard to Guess):
Question: Leo just completed the first draft of a brand identity system for a demanding client. How does he prefer to receive feedback before the next iteration?
Choice A: In a focused 30-minute video call with the creative director to verbally walk through concerns and discuss alternatives in real time.
Choice B: Through a detailed written document with annotated screenshots highlighting specific elements that need revision with suggested directions.
Choice C: In a scheduled one-on-one meeting where he can ask clarifying questions and sketch alternative approaches collaboratively on a whiteboard.
Choice D: Via an asynchronous voice memo from the client paired with marked-up PDF exports he can review independently at his own pace.
Choice E: During a structured team critique session where multiple designers provide diverse perspectives and vote on preferred design directions.

Strengths: All choices contain approximately 28 words. All follow the same structure of method plus format detail plus interaction detail. All represent legitimate professional feedback approaches that someone might prefer.

**Task:**
Generate exactly {num_questions} QA pairs in the format above for the category: {category} using the Persona Profile. Include the word counts section for each question to verify length matching."""


def categories_prompt(persona: str, num_categories: int) -> str:
    """Build the prompt that extracts question categories from a persona profile.

    Args:
        persona: Full persona profile text.
        num_categories: Number of categories to request.

    Returns:
        The prompt text.
    """
    return CATEGORIES_TEMPLATE.format(persona=persona, num_categories=num_categories)


def topics_prompt(persona: str, num_topics: int) -> str:
    """Build the prompt that lists one-sentence chat topics for a persona.

    Args:
        persona: Full persona profile text.
        num_topics: Number of topics to request.

    Returns:
        The prompt text.
    """
    return TOPICS_TEMPLATE.format(persona=persona, num_topics=num_topics)


def chat_prompt(persona: str, topic: str) -> str:
    """Build the prompt that writes one two-turn chat for a topic.

    Args:
        persona: Full persona profile text.
        topic: One chat topic produced by the topics prompt.

    Returns:
        The prompt text.
    """
    return CHAT_TEMPLATE.format(persona=persona, topic=topic)


def questions_prompt(persona: str, category: str, num_questions: int) -> str:
    """Build the prompt that writes evaluation questions for one category.

    Args:
        persona: Full persona profile text.
        category: One category produced by the categories prompt.
        num_questions: Number of questions to request.

    Returns:
        The prompt text.
    """
    return QUESTIONS_TEMPLATE.format(persona=persona, category=category, num_questions=num_questions)
