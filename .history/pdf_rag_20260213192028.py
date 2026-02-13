from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
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
        results = self.search(question, k=3
