Build steps for all three projects

All three projects follow the same 6 steps. Only the knowledge base changes.

Step 1: Make the knowledge base

Theory: A chatbot can only answer what it has stored. This stored data is called the knowledge base.
Do: Write a list of questions, each with an answer and some synonym words (tags).
Example: "What is a stack?" → "A stack follows Last In First Out..."

Step 2: Clean the text (preprocessing)

Theory: Humans write messy text, such as capital letters, punctuation and filler words. Computers need clean, consistent text. This is the first stage of every NLP system.
Do: Apply four small steps to every sentence.

Lowercase: “Stack” becomes “stack”.
Tokenize: split the sentence into words.
Remove stopwords: drop common words like “is”, “the”, “a”.
Lemmatize: reduce words to their base form, like “stacks” to “stack”.

Example: "What is a Stack?" → ["stack"]

Step 3: Turn words into numbers (TF-IDF)

Theory: Computers cannot compare words, only numbers. TF-IDF gives each word a score. A word gets a high score if it appears often in one sentence but rarely in the others, so it is important.
Do: Convert every stored question into a list of numbers (a vector).

Step 4: Find the best answer (cosine similarity)

Theory: Every sentence is now a vector. Cosine similarity measures how close two vectors are, from 0 (nothing in common) to 1 (same meaning).
Do: Convert the user’s question to a vector, compare it with all stored questions, and pick the highest score.
Example: a score of 0.8 is a good match. A score of 0.05 is not a match.

Step 5: Handle special cases (intent detection)

Theory: Intent means what the user wants. “Hello” wants a greeting, not an answer from the knowledge base.
Do:

Check for greeting, goodbye and thanks first.
If the match score is too low, give a polite “I do not know” reply instead of a wrong answer.
Step 6: Add voice

Theory: A voice assistant has three stages: speech-to-text, then NLP, then text-to-speech. We built the NLP part in steps 1 to 5.
Do: Use the browser’s built-in speech recognition for the mic and speech synthesis for the spoken reply.

What changes in each project
AI Tutor First Aid Assistant School Helpdesk
Step 1: Knowledge base DSA and Java questions First aid situations with steps Fees, timings, admission, holidays
Step 5: Special case Greetings and fallback Emergency banner for words like “not breathing” Related question suggestions
Output Short answer Numbered steps and a warning Bullet points and a note
One-line summary for students

Clean the text, convert it to numbers, find the closest stored question, and speak the answer.
