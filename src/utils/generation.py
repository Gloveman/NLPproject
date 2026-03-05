import re

def get_summary(query, context, pipe):
    TOKEN_THRESHOLD = 200
    if len(context) < TOKEN_THRESHOLD:
        return context
    summary_messages = [
        {
            "role": "system",
            "content": (
                "You are an expert research assistant. "
                "Your task is to extract and synthesize information relevant to the user's question from the provided text snippets."
            )
        },
        {
            "role": "user",
            "content": f"""
### Instruction:
Read the following 'Context' and synthesize the information to help answer the 'Question'.
1. **Focus only on relevant facts**: Ignore information that does not help answer the question.
2. **Be concise**: Do not write a long essay. Just provide the key facts in a coherent paragraph.

### Question:
{query}

### Context:
{context}

### Synthesized Summary:
"""
        }
    ]
    outputs = pipe(
        summary_messages,
        max_new_tokens=256,
        pad_token_id=pipe.tokenizer.eos_token_id,
        temperature=0.3
    )
    return outputs[0]["generated_text"][-1]['content']


def generate_answer_few_shot(query, summary, pipe):
    # 1. System Prompt
    system_prompt = (
        "You are a precise QA agent. "
        "Your goal is to answer the user's question based EXCLUSIVELY on the provided context summary. "
    )

    # 2. Few-Shot Examples
    example_dialogues = [
        {"role": "user", "content": """
### Instruction:
Answer the question based on the summary.

### Summary:
The British Army's infantry rifle regiment that Talaiasi Labalaba served in was first created in 1881 by the amalgamation of the 83rd \
         (County of Dublin) Regiment of Foot and the 86th (Royal County Down) Regiment of Foot, becoming the Royal Irish Rifles.

### Question:
The infantry rifle regiment of the British Army that Talaiasi Labalaba served in was first created in what year?
"""},
        {"role": "assistant", "content": "**Final Answer:** 1881"},
        {"role": "user", "content": """
### Instruction:
Answer the question based on the summary.

### Summary:
The European Cup was won by Juventus in 1985, but the season was marred by the Heysel Stadium disaster, \
         in which 39 people were killed and 600 injured, primarily due to a violent confrontation between Juventus and Liverpool fans.

### Question:
Who was the winner of the European Cup the year that 39 people were killed?
"""},
        {"role": "assistant", "content": "**Final Answer:** Juventus"}
    ]

    # 3. Actual Task
    actual_task_message = {
        "role": "user",
        "content": f"""
### Instruction:
Now, answer the following NEW question based on the NEW summary provided below.
Do not use the previous examples' answers.

### Summary:
{summary}

### Question:
{query}
"""
    }

    # 4. Assemble messages
    answer_messages = [{"role": "system", "content": system_prompt}] + example_dialogues + [actual_task_message]

    # 5. Generate Output
    outputs = pipe(
        answer_messages,
        max_new_tokens=128,
        pad_token_id=pipe.tokenizer.eos_token_id,
        temperature=0.1
    )

    full_output = outputs[0]["generated_text"][-1]['content']

    # 6. Regex match answer
    match_first = re.search(r"\*\*Final Answer:\*\*\s*(.*)", full_output, re.DOTALL | re.IGNORECASE)
    match_second = re.search(r"Final Answer:\s*(.*)", full_output, re.DOTALL | re.IGNORECASE)

    if match_first:
        return match_first.group(1).strip()
    if match_second:
        return match_second.group(1).strip()
    return full_output
