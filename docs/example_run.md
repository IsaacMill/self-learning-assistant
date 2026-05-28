# Example CLI Run

```text
$ python -m assistant.chat
Self-learning assistant MVP. Type /quit to exit, /debug to toggle memory debug.

You: /debug
Debug mode: True

You: Remember that I prefer concise answers.

Assistant: I am running in local fallback mode without an LLM API key. I can still log, retrieve, judge, and reflect on this message: Remember that I prefer concise answers.

Judges:
- factuality: pass=True score=0.80 No obvious unsupported factuality issue detected.
- safety: pass=True score=1.00 No blocked action or secret detected.
- usefulness: pass=True score=0.80 Answer appears responsive.
- consistency: pass=True score=0.80 No memories were used, so consistency risk is low.

Possible mistakes:
- Answer quality is limited because no LLM API key was configured.

Proposed trusted memory: i prefer concise answers
Store as trusted semantic memory? [y/N]: y
Stored memory 6bfb6b1d.

You: How should you answer me?

Retrieved memories:
- 6bfb6b1d score=0.35: i prefer concise answers

Assistant: I am running in local fallback mode without an LLM API key. Relevant memory context:
Relevant memories:
- [semantic/trusted] i prefer concise answers

For your message: How should you answer me?
```
