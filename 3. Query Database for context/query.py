import openai
import pinecone

# Credentials
OPENAI_API_KEY = "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"
EMBEDDINGS_MODEL = "text-embedding-ada-002"

PINECONE_API_KEY = "d22a004e-e747-4ffb-b5b5-b39da101eb2c"
PINECONE_ENVIRONMENT = "gcp-starter"
PINECONE_INDEX = "example-index"

# Set up Pinecone
pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_ENVIRONMENT)

if PINECONE_INDEX not in pinecone.list_indexes():
    # Create the index
    pinecone.create_index(PINECONE_INDEX, dimension=1536, metric="cosine")

# Get the index
index = pinecone.Index(PINECONE_INDEX)

# Set OpenAI API key
openai.api_key = OPENAI_API_KEY

def vectorize(text):
    # Vectorize text
		response = openai.embeddings.create(
				input=[text],
				model=EMBEDDINGS_MODEL,
		)

		return response.data[0].embedding

def query(vector):
		# Query Pinecone
		results = index.query(vector, top_k=5, include_metadata=True)
		return results

QUERY = "What is kubernetes?"

# Vectorize the query
vector = vectorize(QUERY)

# Query Pinecone
q = query(vector)

matches = q["matches"]

context = []

for match in matches:
		context.append(match["metadata"]["text"] + "\n")
		context.append("\n\n----------\n\n")

print("Context: \n" + "".join(context))
print(len(context))

# Build the prompt
prompt = "Context: \n" + "".join(context) + "\n" + "Query: " + QUERY + "\nAnswer:"

# Save to file query.txt
with open("query.txt", "w") as f:
		f.write(prompt)
