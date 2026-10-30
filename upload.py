
import os
from dotenv import load_dotenv
from pinecone import Pinecone

load_dotenv()

pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])

assistant = pc.assistants.describe(name="chat")
print("Assistant:", assistant.name)
print("Host:", assistant.host)

# Replace with the real path to your document
file = pc.assistants.upload_file(
    assistant_name="chat",
    file_path="knowledge.pdf"
)

print("File status:", file.status)
