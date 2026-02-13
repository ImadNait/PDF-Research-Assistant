from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter 
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
import os
class PDFResearchAssistant:
    def __init__(self, pdf_directory="pdfs"):
        self.pdf_directory = pdf_directory
        self.embeddings = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2"
        )
        self.vectorstore = None
        
        # Create directory if it doesn't exist
        os.makedirs(pdf_directory, exist_ok=True)
    
    def load_pdfs(self):
        """Load all PDFs from directory"""
        print(f"📚 Loading PDFs from {self.pdf_directory}/...")
        
        loader = DirectoryLoader(
            self.pdf_directory,
            glob="**/*.pdf",
            loader_cls=PyPDFLoader
        )
        
        documents = loader.load()
        print(f"✅ Loaded {len(documents)} pages from PDFs")
        return documents
    
    def process_documents(self, documents):
        """Split documents into chunks"""
        print("✂️  Splitting documents into chunks...")
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        
        chunks = text_splitter.split_documents(documents)
        print(f"✅ Created {len(chunks)} chunks")
        return chunks
    
    def create_vectorstore(self, chunks):
        """Create vector database"""
        print("🔮 Creating vector database...")
        
        self.vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=self.embeddings,
            persist_directory="./pdf_db"
        )
        
        print("✅ Vector database ready!")
    
    def load_existing(self):
        """Load existing database"""
        print("📂 Loading existing database...")
        self.vectorstore = Chroma(
            persist_directory="./pdf_db",
            embedding_function=self.embeddings
        )
        print("✅ Database loaded!")
    
    def setup(self):
        """One-time setup"""
        documents = self.load_pdfs()
        if not documents:
            print("⚠️  No PDFs found! Add some .pdf files to the 'pdfs/' folder")
            return False
        
        chunks = self.process_documents(documents)
        self.create_vectorstore(chunks)
        return True
    
    def search(self, query, k=4):
        """Search for relevant information"""
        if not self.vectorstore:
            print("❌ No database loaded!")
            return []
        
        print(f"\n🔍 Searching: '{query}'")
        print("="*60)
        
        results = self.vectorstore.similarity_search(query, k=k)
        
        for i, doc in enumerate(results, 1):
            source = doc.metadata.get('source', 'Unknown')
            page = doc.metadata.get('page', '?')
            
            print(f"\n📄 Result {i}:")
            print(f"Source: {os.path.basename(source)} (Page {page})")
            print(f"Content:\n{doc.page_content[:300]}...")
            print("-"*60)
        
        return results
    
    def ask(self, question):
        """Ask a question and get contextual answer"""
        results = self.search(question, k=3)
        
        if not results:
            print("❌ No relevant information found.")
            return
        
        print("\n💡 Summary from your PDFs:")
        print("="*60)
        
        # Combine contexts
        context = "\n\n".join([doc.page_content for doc in results])
        
        print(f"\nBased on the documents:")
        print(context[:800] + "...\n")
        
        # Show sources
        sources = set()
        for doc in results:
            source = os.path.basename(doc.metadata.get('source', 'Unknown'))
            page = doc.metadata.get('page', '?')
            sources.add(f"{source} (p.{page})")
        
        print(f"📚 Sources: {', '.join(sources)}")


def main():
    print("📖 PDF Research Assistant")
    print("="*60)
    
    assistant = PDFResearchAssistant()
    
    # Check for existing database
    if os.path.exists("./pdf_db"):
        print("📊 Existing database found!")
        choice = input("Load existing (L) or rebuild (R)? [L/R]: ").strip().upper()
        
        if choice == 'L':
            assistant.load_existing()
        else:
            if not assistant.setup():
                return
    else:
        print("🆕 First time setup")
        print(f"Add PDF files to the 'pdfs/' folder first!")
        
        if not os.listdir(assistant.pdf_directory):
            print("\n⚠️  'pdfs/' folder is empty!")
            print("Add some PDFs and run again.")
            return
        
        if not assistant.setup():
            return
    
    # Interactive mode
    print("\n" + "="*60)
    print("🎯 Ask questions about your PDFs!")
    print("Type 'quit' to exit")
    print("="*60)
    
    while True:
        question = input("\n❓ Your question: ").strip()
        
        if question.lower() in ['quit', 'exit', 'q']:
            print("👋 Goodbye!")
            break
        
        if not question:
            continue
        
        assistant.ask(question)


if __name__ == "__main__":
    main()