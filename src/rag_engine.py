from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


def load_knowledge_base(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        text = file.read()

    return text


def split_into_chunks(text):
    sections = text.split("\n\n")
    chunks = []

    for section in sections:
        section = section.strip()

        # Skip empty sections
        if not section:
            continue

        # Skip the document title
        if section == "AML Transaction Red Flags":
            continue

        chunks.append(section)

    return chunks



def create_embeddings(chunks):
    model = SentenceTransformer("all-MiniLM-L6-v2")

    embeddings = model.encode(chunks)

    return model,embeddings


def search_knowledge_base(query, chunks, embeddings, model, top_k=3):
    query_embedding = model.encode([query])

    similarities = cosine_similarity(
        query_embedding,
        embeddings
    )[0]

    
    top_indices = similarities.argsort()[::-1][:top_k]

    results = []

    for index in top_indices:
        results.append({
            "chunk": chunks[index],
            "similarity": float(similarities[index])
        })

    return results


if __name__ == "__main__":

    knowledge = load_knowledge_base(
        "knowledge_base/aml_red_flags.txt"
    )

    chunks = split_into_chunks(knowledge)

    model, embeddings = create_embeddings(chunks)

    query = "What are the risks of an unusual large transaction?"

    results= search_knowledge_base(
        query,
        chunks,
        embeddings,
        model,
        top_k=3
    )

    print("Query:")
    print(query)

    print("\nTop Retrieved Results:")

    for i, result in enumerate(results, start=1):
        print(f"\nResult {i}")
        print(result["chunk"])
        print(
            "Similarity:",
            round(result["similarity"], 4)
        )