from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaLLM
from langchain.chains import RetrievalQA
import os


class PDFResearchAssistant:
    def __init__(self, pdf_directory="pdfs"):
        self.pdf_directory = pdf_directory
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        self.vectorstore = None
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

    def search(self, query, k=4):
        if not self.vectorstore:
            print("No database loaded!")
            return []

        print(f"\nSearching: '{query}'")
        print("=" * 60)
        results = self.vectorstore.similarity_search(query, k=k)

        for i, doc in enumerate(results, 1):
            source = doc.metadata.get("source", "Unknown")
            page = doc.metadata.get("page", "?")
            print(f"\nResult {i}:")
            print(f"Source: {os.path.basename(source)} (Page {page})")
            print(f"Content:\n{doc.page_content[:300]}...")
            print("-" * 60)

        return results

    def ask(self, question):
        results = self.search(question, k=3)

        if not results:
            print("No relevant information found.")
            return

        print("\nSummary from your PDFs:")
        print("=" * 60)
        context = "\n\n".join([doc.page_content for doc in results])

        print("\nBased on the documents:")
        print(context[:800] + "...\n")

        sources = set()
        for doc in results:
            source = os.path.basename(doc.metadata.get("source", "Unknown"))
            page = doc.metadata.get("page", "?")
            sources.add(f"{source} (p.{page})")

        print(f"Sources: {', '.join(sources)}")

    def ask_with_llm(self, question):
        if not self.vectorstore:
            print("No database loaded!")
            return

        print(f"\nGenerating answer for: '{question}'")
        print("=" * 60)

        llm = OllamaLLM(model="llama2", temperature=0.3)

        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=self.vectorstore.as_retriever(search_kwargs={"k": 3}),
            return_source_documents=True,
        )

        print("Thinking...")
        result = qa_chain.invoke({"query": question})

        print(f"\nAnswer:\n{result['result']}\n")

        print("Sources:")
        for i, doc in enumerate(result["source_documents"], 1):
            source = os.path.basename(doc.metadata.get("source", "Unknown"))
            page = doc.metadata.get("page", "?")
            print(f"  {i}. {source} (Page {page})")

        print("=" * 60)


def main():
    print("PDF Research Assistant")
    print("=" * 60)

    assistant = PDFResearchAssistant()

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

        if not os.path.exists(assistant.pdf_directory) or not os.listdir(
            assistant.pdf_directory
        ):
            print("\n'pdfs/' folder is empty!")
            print("Add some PDFs and run again.")
            return

        if not assistant.setup():
            return

    print("\n" + "=" * 60)
    print("Choose mode:")
    print("1. Basic Search (shows relevant chunks)")
    print("2. AI Answer (uses Ollama LLM)")
    mode = input("Enter 1 or 2 [1]: ").strip() or "1"

    print("\n" + "=" * 60)
    print("Ask questions about your PDFs!")
    print("Type 'quit' to exit, 'switch' to change mode")
    print("=" * 60)

    while True:
        question = input("\nYour question: ").strip()

        if question.lower() in ["quit", "exit", "q"]:
            print("Goodbye!")
            break

        if question.lower() == "switch":
            mode = "2" if mode == "1" else "1"
            print(
                f"Switched to {'AI Answer' if mode == '2' else 'Basic Search'} mode"
            )
            continue

        if not question:
            continue

        if mode == "2":
            assistant.ask_with_llm(question)
        else:
            assistant.ask(question)


if __name__ == "__main__":
    main()