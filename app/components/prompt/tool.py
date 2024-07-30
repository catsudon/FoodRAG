HALLUCINATION_GRADER_PROMPT_TEMPLATE = """
You are a grader assessing whether an answer is grounded in / supported by a set of facts. Give a binary 'yes' or 'no' `isHallucinate` to indicate
whether the answer is grounded in / supported by a set of facts provided. Provide the binary `isHallucinate` as a JSON with two keys `isHallucinate`
and `reason` and no preamble or explanation. You must provide a reason for your decision. Check if the answer is grounded in the facts provided.
Here are the facts:
\n ------- \n
{documents}
\n ------- \n
Here is the answer: {generation}"""


TEXT_REVISER_PROMPT_TEMPLATE = """
As an mute assistant, your aim is to enhance text and table readability based on specific guidelines and previous context:
Reframe sentences and sections for better comprehension.
Eliminate unclear text, e.g., content with excessive symbols or gibberish.
Shorten text without losing information. Suggestion: Summarize lengthy phrases where possible.
Rectify poorly formatted tables, e.g., adjust column alignment for clarity.
Preserve clear, understandable text as is. Example: "Use direct and easily comprehensible sentences."
Refrain from responding if text is entirely unclear or ambiguous, e.g., incomprehensible or garbled content.
Remove standalone numbers or letters not associated with text, e.g., isolated digits or letters lacking context.
Remove redundant or repetitive text, e.g., content that is repeated or reiterated unnecessarily. Suggestion: Combine repetitive sentences to previous context if possible.
Exclude non-factual elements like selection marks, addresses, picture marks, or drawings.
Ensure modifications maintain the original text's clarity and information conveyed.
Remove any irrelevant or unnecessary information. Like addresses, phone numbers, or any advertisement purpose information.
Answer back the revised text without additional comments before or after, avoiding comments about how or which guidelines have been followed.
Answer without any header or footer, only the revised text.
The context of previous text is as follows:
{previous_context}

-------------------------------------------------------------------------------------------------

Please revise the following text based on these guidelines no notes or comments:
Text: {context}

Revised text follow this line:
"""

TEXT_NO_PREV_CONTEXT_REVISER_PROMPT_TEMPLATE = """
As an mute assistant, your aim is to enhance text and table readability based on specific guidelines:
Reframe sentences and sections for better comprehension.
Eliminate unclear text, e.g., content with excessive symbols or gibberish.
Shorten text without losing information. Suggestion: Summarize lengthy phrases where possible.
Rectify poorly formatted tables, e.g., adjust column alignment for clarity.
Preserve clear, understandable text as is. Example: "Use direct and easily comprehensible sentences."
Refrain from responding if text is entirely unclear or ambiguous, e.g., incomprehensible or garbled content.
Remove standalone numbers or letters not associated with text, e.g., isolated digits or letters lacking context.
Remove redundant or repetitive text, e.g., content that is repeated or reiterated unnecessarily.
Exclude non-factual elements like selection marks, addresses, picture marks, or drawings.
Ensure modifications maintain the original text's clarity and information conveyed.
Remove any irrelevant or unnecessary information. Like addresses, phone numbers, or any advertisement purpose information.
Answer back the revised text without any additional comments before or after, avoiding comments about how or which guidelines have been followed!!
Answer without any header or footer, only the revised text.
Please revise the following text based on these guidelines no notes or comments:
Text: {context}

Revised text follow this line:
"""


ANSWER_GENERATOR_PROMPT_TEMPLATE = """
You are an assistant for question-answering tasks.
Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know.
Use five sentences maximum and keep the answer concise
Question: {question}
Context: {context}
Just put the answer below this line:"""


ANSWER_GRADER_PROMPT_TEMPLATE = """
You are a grader assessing whether an answer is useful to resolve a question. Give a binary `isUseful` true or false to indicate whether the answer is
useful to resolve a question. Provide the binary `isUseful` as a JSON with two keys `isUseful`, `reason` and no preamble or explanation.
Here is the answer:
\n ------- \n
{generation}
\n ------- \n
Here is the question: {question}\n"""


CONTEXT_TRANSLATOR_PROMPT_TEMPLATE = """
You are an assistant for translating context.
Text back the revised text that adapted to language without any additional comments before or after.
Answer without any header or footer, only the revised text.
The text should be natural and fluent in the target language.

The target language is : {language}

Here is the original text:
{original}

-------------------------------------------------------------------------------------------------
Here is the translated text you might need to revise:
{translated}
"""
