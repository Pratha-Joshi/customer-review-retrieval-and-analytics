from __future__ import annotations
def evidence_prompt(question: str, results: list) -> str:
    evidence='\n\n'.join(f"[Review {r.document['metadata']['review_id']}] rating={r.document['metadata']['rating']}\n{r.document['text']}" for r in results)
    return f"""Answer the question using only the review evidence below. Cite each claim as [Review ID]. State when evidence is insufficient, distinguish inference from evidence, and acknowledge material conflicts. Do not use outside knowledge.\n\nQuestion: {question}\n\nEvidence:\n{evidence}\n\nAnswer:"""
def citations(results): return [{"review_id":r.document['metadata']['review_id'], "document_id":r.document['id']} for r in results]
def fallback_answer(results):
    if not results: return "Insufficient evidence: no reviews matched the request."
    return "Local generation is unavailable. Inspect the cited evidence; no unsupported synthesis was generated."
