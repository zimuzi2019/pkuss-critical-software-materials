"""Vanilla（COT）：LLM被要求简单地输出其偏好，而没有任何解释，以零镜头的方式提示LLM。

Chain-of-Thoughts：指示LLM首先生成一个简明的推理，然后在两个输出之间生成其偏好。

Self-Generated Reference（Reference）：提示LLM evaluator生成一个给定指令I的输出。然后，在进行比较时，将生成的输出作为参考传递给LLM evaluator。

ChatEval：多个LLM评估者，通过不同的角色提示进行个性化。

Rules（*）：列出了LLM评估器在进行比较时要遵循的一些一般规则（规则几乎普遍地提高了评估者的准确性）。"""
def build_message(system, s):
    message = [
        {'role': 'system', 'content': system},
        {"role": "user", "content": s},
    ]

    return message

def prompt1(question, answer, ref=None, mode=0, isMTBench=False):
    if mode == 0:  # pairwise
        system = """[System]
Please act as an impartial judge and evaluate the quality of the responses provided by two AI assistants to the user question displayed below. Your evaluation should consider factors such as the helpfulness, relevance, accuracy, depth, creativity, and level of detail of the response. Begin your evaluation by comparing the two responses and provide a short explanation. Avoid any position biases and ensure that the order in which the responses were presented does not influence your decision. Do not allow the length of the responses to influence your evaluation. Do not favor certain names of the assistants. Be as objective as possible. After providing your explanation, output your final verdict by strictly following this format: "[[A]]" if assistant A is better, "[[B]]" if assistant B is better, and "[[C]]" for a tie."""
        if isMTBench:
            system += "Notes: The Q&A between user and AI assistants are all Multi-turn Conversation "
        s=f"""[User Question]
        {question}
        [The Start of Assistant A's Answer]
        {answer[0]}
        [The End of Assistant A's Answer]
        [The Start of Assistant B's Answer]
        {answer[1]}
        [The End of Assistant B's Answer]
        """ if not isMTBench else f"""
[The Start of Assistant A's Conversation]
{answer[0]}
[The End of Assistant A's Conversation]
[The Start of Assistant B's Conversation]
{answer[1]}
[The End of Assistant B's Conversation]"""

    elif mode == 1:  # pairwise with reference
        system = """[System]
Please act as an impartial judge and evaluate the quality of the responses provided by two AI assistants to the user question displayed below. Your evaluation should consider correctness and helpfulness. You'll be given a reference answer, Assistant A's answer, and assistant B's answer. Your job is to evaluate which assistant's answer is better. Begin your evaluation by comparing both assistants' answers with the reference answer. Identify and correct any mistakes. Avoid any position biases and ensure that the order in which the responses were presented does not influence your decision. Do not allow the length of the responses to influence your evaluation. Do not favor certain names of the assistants. Be as objective as possible. After providing your explanation, output your final verdict by strictly following this format: "[[A]]" if assistant A is better, "[[B]]" if assistant B is better, and "[[C]]" for a tie."""
        s=f"""[User Question]
        {question}
        [The Start of Reference Answer]
        {ref}
        [The End of Reference Answer]

        [The Start of Assistant A's Answer]
        {answer[0]}
        [The End of Assistant A's Answer]

        [The Start of Assistant B's Answer]
        {answer[1]}
        [The End of Assistant B's Answer]"""
    else:  # pointwise
        system = """[System]
Please act as an impartial judge and evaluate the quality of the response provided by an AI assistant to the user question displayed below. Your evaluation should consider factors such as the helpfulness, relevance, accuracy, depth, creativity, and level of detail of the response. Begin your evaluation by providing a short explanation. Be as objective as possible. After providing your explanation, please rate the response on a scale of 1 to 10 by strictly following this format: "[[rating]]", for example: "Rating: [[5]]". """

        s = f"""[Question]
            {question}
            [The Start of Assistant's Answer]
            {answer[0]}
            [The End of Assistant's Answer]"""
    return build_message(system, s), [ '[[C]]', '[[A]]', '[[B]]' ]


def prompt1_cn(question, answer, ref=None, mode=0, isMTBench=False):
    if mode == 0:  # pairwise
        system = """[System]
作为一个公正的评委,请评估两个AI助手对下方显示的用户问题所提供的响应的质量。你的评估应考虑回应的帮助程度、相关性、准确性、深度、创造性和细节程度等因素。通过比较两个回应开始你的评估,并提供一个简短的解释。避免任何立场偏见,确保回应的呈现顺序不会影响你的决定。不要让回应的长度影响你的评估。不要偏爱某些助手的名字。尽可能做到客观。注意：务必在提供解释以后才给出判断。在提供解释后,严格按照以下格式输出你的最终判决:"[[A]]" 如果助手A更好,"[[B]]" 如果助手B更好,"[[C]]" 表示平局。"""

        s=f"""[User Question]
{question}
[The Start of Assistant A's Answer]
{answer[0]}
[The End of Assistant A's Answer]
[The Start of Assistant B's Answer]
{answer[1]}
[The End of Assistant B's Answer]
""" if not isMTBench else f"""
[The Start of Assistant A's Conversation]
{answer[0]}
[The End of Assistant A's Conversation]
[The Start of Assistant B's Conversation]
{answer[1]}
[The End of Assistant B's Conversation]"""
    elif mode == 1:  # pairwise with reference
        system = """[System]
请作为一名公正的评委,评估两位AI助手对下方显示的用户问题所提供的回复质量。你的评估应考虑正确性和有用性。你将获得一个参考答案、助手A的答案,以及助手B的答案。你的工作是评估哪个助手的回答更好。通过将两位助手的答案与参考答案进行比较,开始你的评估。识别并纠正任何错误。避免任何立场偏见,确保回复的呈现顺序不会影响你的决定。不要让回复的长度影响你的评估。不要偏爱某些助手的名字。尽可能做到客观。注意：务必在提供解释以后才给出判断。在提供解释后,严格按照以下格式输出你的最终判决:"[[A]]" 如果助手A更好,"[[B]]"如果助手B更好,"[[C]]" 表示平局。"""
        s=f"""[User Question]
        {question}
        [The Start of Reference Answer]
        {ref}
        [The End of Reference Answer]

        [The Start of Assistant A's Answer]
        {answer[0]}
        [The End of Assistant A's Answer]

        [The Start of Assistant B's Answer]
        {answer[1]}
        [The End of Assistant B's Answer]"""
    else:  # pointwise
        system = """[System]
作为一名公正的评委,请评估下方所示用户问题由一位AI助手提供的回答质量。你的评估应该考虑回复的帮助程度、相关性、准确性、深度、创造力和细节程度等因素。从提供一个简短的解释开始你的评估。尽量做到客观可能的。注意：务必在提供解释以后才给出判断。在提供解释后,请严格按照以下格式对回复进行评分,评分范围为1至10:"[[rating]]",例如:"评分:[[5]]"。 """

        s = f"""[Question]
            {question}
            [The Start of Assistant's Answer]
            {answer[0]}
            [The End of Assistant's Answer]"""
    return build_message(system, s), [ '[[C]]', '[[A]]', '[[B]]' ]


def prompt_vanila(question, answer, ref=None, mode=0, isMTBench=False):
    if mode == 0:  # pairwise
        system = """[System]
Please act as an impartial judge and evaluate the quality of the responses provided by two AI assistants to the user question displayed below. Select the Assistant A's answer or Assistant B's answer that is better for the given instruction. Do NOT provide any explanation for your choice. Output your final verdict by strictly following this format: "[[A]]" if assistant A is better, "[[B]]" if assistant B is better, and "[[C]]" for a tie."""
        if isMTBench:
            system += "Notes: The Q&A between user and AI assistants are all Multi-turn Conversation "
        s=f"""[User Question]
        {question}
        [The Start of Assistant A's Answer]
        {answer[0]}
        [The End of Assistant A's Answer]
        [The Start of Assistant B's Answer]
        {answer[1]}
        [The End of Assistant B's Answer]
        """ if not isMTBench else f"""
[The Start of Assistant A's Conversation]
{answer[0]}
[The End of Assistant A's Conversation]
[The Start of Assistant B's Conversation]
{answer[1]}
[The End of Assistant B's Conversation]"""

    return build_message(system, s), [ '[[C]]', '[[A]]', '[[B]]' ]

def prompt_CoT(question, answer, ref=None, mode=0, isMTBench=False):
    if mode == 0:  # pairwise
        system = """[System]
Please act as an impartial judge and evaluate the quality of the responses provided by two AI assistants to the user question displayed below. Begin your evaluation by comparing the two responses and provide a short explanation. After providing your explanation, output your final verdict by strictly following this format: "[[A]]" if assistant A is better, "[[B]]" if assistant B is better, and "[[C]]" for a tie."""
        if isMTBench:
            system += "Notes: The Q&A between user and AI assistants are all Multi-turn Conversation "
        s=f"""[User Question]
        {question}
        [The Start of Assistant A's Answer]
        {answer[0]}
        [The End of Assistant A's Answer]
        [The Start of Assistant B's Answer]
        {answer[1]}
        [The End of Assistant B's Answer]
        """ if not isMTBench else f"""
[The Start of Assistant A's Conversation]
{answer[0]}
[The End of Assistant A's Conversation]
[The Start of Assistant B's Conversation]
{answer[1]}
[The End of Assistant B's Conversation]"""

    return build_message(system, s), [ '[[C]]', '[[A]]', '[[B]]' ]

def prompt_Swap_CoT(question, answer, ref=None, mode=0, isMTBench=False):
    if mode == 0:  # pairwise
        system = """[System]
Please act as an impartial judge and evaluate the quality of the responses provided by two AI assistants to the user question displayed below. Your evaluation should consider factors such as the helpfulness, relevance, accuracy, depth, creativity, and level of detail of the response. Begin your evaluation by comparing the two responses and provide a short explanation. Avoid any position biases and ensure that the order in which the responses were presented does not influence your decision. Do not allow the length of the responses to influence your evaluation. Do not favor certain names of the assistants. Be as objective as possible. After providing your explanation, output your final verdict by strictly following this format: "[[A]]" if assistant A is better, "[[B]]" if assistant B is better, and "[[C]]" for a tie."""
        if isMTBench:
            system += "Notes: The Q&A between user and AI assistants are all Multi-turn Conversation "
        s=f"""[User Question]
        {question}
        [The Start of Assistant A's Answer]
        {answer[0]}
        [The End of Assistant A's Answer]
        [The Start of Assistant B's Answer]
        {answer[1]}
        [The End of Assistant B's Answer]
        """ if not isMTBench else f"""
[The Start of Assistant A's Conversation]
{answer[0]}
[The End of Assistant A's Conversation]
[The Start of Assistant B's Conversation]
{answer[1]}
[The End of Assistant B's Conversation]"""

    return build_message(system, s), [ '[[C]]', '[[A]]', '[[B]]' ]


def prompt_vanila_rule(question, answer, ref=None, mode=0, isMTBench=False):
    if mode == 0:  # pairwise
        system = """[System]
Please act as an impartial judge and evaluate the quality of the responses provided by two AI assistants to the user question displayed below. Select the Assistant A's answer or Assistant B's answer that is better for the given instruction. Do NOT provide any explanation for your choice. Output your final verdict by strictly following this format: "[[A]]" if assistant A is better, "[[B]]" if assistant B is better, and "[[C]]" for a tie.

Here are some rules of the evaluation:
(1) You should prioritize evaluating whether the answer honestly/precisely/closely executes the instruction, then consider its helpfulness, accuracy, level of detail, harmlessness, etc.
(2) Answers should NOT contain more/less than what the question asks for, as such answers do NOT precisely answer the question.
(3) Avoid any position biases and ensure that the order in which the responses were presented does not influence your decision, as Answer A and Answer B are **equally likely** to be the better.
(4) Do not allow the length of the responses to influence your evaluation.
(5) Be as objective as possible."""
        if isMTBench:
            system += "Notes: The Q&A between user and AI assistants are all Multi-turn Conversation "
        s=f"""[User Question]
        {question}
        [The Start of Assistant A's Answer]
        {answer[0]}
        [The End of Assistant A's Answer]
        [The Start of Assistant B's Answer]
        {answer[1]}
        [The End of Assistant B's Answer]
        """ if not isMTBench else f"""
[The Start of Assistant A's Conversation]
{answer[0]}
[The End of Assistant A's Conversation]
[The Start of Assistant B's Conversation]
{answer[1]}
[The End of Assistant B's Conversation]"""

    return build_message(system, s), [ '[[C]]', '[[A]]', '[[B]]' ]


def prompt_CoT_rule(question, answer, ref=None, mode=0, isMTBench=False):
    if mode == 0:  # pairwise
        system = """[System]
Please act as an impartial judge and evaluate the quality of the responses provided by two AI assistants to the user question displayed below. Begin your evaluation by comparing the two responses and provide a short explanation. After providing your explanation, output your final verdict by strictly following this format: "[[A]]" if assistant A is better, "[[B]]" if assistant B is better, and "[[C]]" for a tie.

Here are some rules of the evaluation:
(1) You should prioritize evaluating whether the answer honestly/precisely/closely executes the instruction, then consider its helpfulness, accuracy, level of detail, harmlessness, etc.
(2) Answers should NOT contain more/less than what the question asks for, as such answers do NOT precisely answer the question.
(3) Avoid any position biases and ensure that the order in which the responses were presented does not influence your decision, as Answer A and Answer B are **equally likely** to be the better.
(4) Do not allow the length of the responses to influence your evaluation.
(5) Be as objective as possible."""
        if isMTBench:
            system += "Notes: The Q&A between user and AI assistants are all Multi-turn Conversation "
        s=f"""[User Question]
        {question}
        [The Start of Assistant A's Answer]
        {answer[0]}
        [The End of Assistant A's Answer]
        [The Start of Assistant B's Answer]
        {answer[1]}
        [The End of Assistant B's Answer]
        """ if not isMTBench else f"""
[The Start of Assistant A's Conversation]
{answer[0]}
[The End of Assistant A's Conversation]
[The Start of Assistant B's Conversation]
{answer[1]}
[The End of Assistant B's Conversation]"""

    return build_message(system, s), [ '[[C]]', '[[A]]', '[[B]]' ]