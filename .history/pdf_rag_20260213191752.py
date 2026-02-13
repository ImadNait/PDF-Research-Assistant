from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import Ollama
import os


class PDFResearchAssistant:
    def __init__(self, pdf_directory="pdfs", llm_model="llama3.1"):
        self.pdf_directory = pdf_directory
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        self.vectorstore = None
        self.llm = Ollama(model=llm_model, temperature=0)
        os.makedirs(pdf_directory, exist_ok=True)

    def load_pdfs(self):
        print(f"Loading PDFs from {self.pdf_directory}/...")
        loader = DirectoryLoader(
            self.pdf_directory,
            glob="**/*.pdf",
            loader_cls=PyPDFLoader,
        )
        documents = loader.load()
        print(f"Loaded {len(documents)} pages from PDFs")
        return documents

    def process_documents(self, documents):
        print("Splitting documents into chunks...")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )
        chunks = text_splitter.split_documents(documents)
        print(f"Created {len(chunks)} chunks")
        return chunks

    def create_vectorstore(self, chunks):
        print("Creating vector database...")
        self.vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=self.embeddings,
            persist_directory="./pdf_db",
        )
        print("Vector database ready!")

    def load_existing(self):
        print("Loading existing database...")
        self.vectorstore = Chroma(
            persist_directory="./pdf_db",
            embedding_function=self.embeddings,
        )
        print("Database loaded!")

    def setup(self):
        documents = self.load_pdfs()
        if not documents:
            print("No PDFs found! Add some .pdf files to the 'pdfs/' folder")
            return False
        chunks = self.process_documents(documents)
        self.create_vectorstore(chunks)
        return True

    def search(self, query, k=4, show_results=True):
        if not self.vectorstore:
            print("No database loaded!")
            return []

        print(f"\nSearching: '{query}'")
        print("=" * 60)
        results = self.vectorstore.similarity_search(query, k=k)

        if show_results:
            for i, doc in enumerate(results, 1):
                source = doc.metadata.get("source", "Unknown")
                page = doc.metadata.get("page", "?")
                print(f"\nResult {i}:")
                print(f"Source: {os.path.basename(source)} (Page {page})")
                print(f"Content:\n{doc.page_content[:300]}...")
                print("-" * 60)

        return results

    def ask(self, question):
        results = self.search(question, k=5, show_results=False)

        if not results:
            print("No relevant information found.")
            return

        context_blocks = []
        source_lines = []

        for i, doc in enumerate(results, 1):
            source = os.path.basename(doc.metadata.get("source", "Unknown"))
            page = doc.metadata.get("page", "?")
            context_blocks.append(f"[{i}] {doc.page_content}")
            source_lines.append(f"[{i}] {source} (p.{page})")

        context = "\n\n".join(context_blocks)

        prompt = (
            "You are a careful research assistant.\n"
            "Use only the provided context to answer the question.\n"
            "If the context is insufficient, say exactly: I don't have enough information in the provided PDFs.\n"
            "Keep the answer concise and factual.\n\n"
            f"Question:\n{question}\n\n"
            f"Context:\n{context}\n\n"
            "Answer:"
        )

        try:
            answer = self.llm.invoke(prompt)
            if not isinstance(answer, str):
                answer = str(answer)
        except Exception as e:
            print(f"LLM error: {e}")
            return

        print("\nAnswer:")
        print("=" * 60)
        print(answer.strip())
        print("\nSources:")
        print(", ".join(source_lines))


def main():
    print("PDF Research Assistant")
    print("=" * 60)

    model_name = input("Ollama model [llama3.1]: ").strip() or "llama3.1"
    assistant = PDFResearchAssistant(llm_model=model_name)

    if os.path.exists("./pdf_db"):
        print("Existing database found!")
        choice = input("Load existing (L) or rebuild (R)? [L/R]: ").strip().upper()

        if choice == "L":
            assistant.load_existing()
        else:
            if not assistant.setup():
                return
    else:
        print("First time setup")
        print("Add PDF files to the 'pdfs/' folder first!")

        if not os.listdir(assistant.pdf_directory):
            print("\n'pdfs/' folder is empty!")
            print("Add some PDFs and run again.")
            return

        if not assistant.setup():
            return

    print("\n" + "=" * 60)
    print("Ask questions about your PDFs!")
    print("Type 'quit' to exit")
    print("=" * 60)

    while True:
        question = input("\nYour question: ").strip()

        if question.lower() in ["quit", "exit", "q"]:
            print("Goodbye!")
            break

        if not question:
            continue

        assistant.ask(question)


if __name__ == "__main__":
    main()
