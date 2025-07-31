import os
import shutil
import re
from collections import Counter, defaultdict
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

def setup_faculty_rag_system():
    """Setup the RAG system with the new faculty data"""
    
    print("🚀 Setting up LNMIIT Faculty RAG System...\n")
    
    # Step 1: Check if faculty_details.txt exists
    if not os.path.exists("faculty_details.txt"):
        print("❌ faculty_details.txt not found!")
        print("Please make sure the file is in the current directory.")
        return False
    
    # Step 2: Copy faculty_details.txt to faculty_data.txt (for compatibility)
    print("📁 Preparing faculty data...")
    shutil.copy("faculty_details.txt", "faculty_data.txt")
    print("✅ Created faculty_data.txt for RAG system")
    
    # Step 3: Load and analyze the data
    try:
        loader = TextLoader("faculty_data.txt", encoding='utf-8')
        documents = loader.load()
        print(f"✅ Loaded faculty data ({len(documents[0].page_content)} characters)")
        
        # Show some stats
        content = documents[0].page_content
        lines = content.split('\n')
        faculty_count = content.count('Designation:')
        print(f"📊 Found {faculty_count} faculty members")
        
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return False
    
    # Step 4: Split text into chunks
    print("✂️ Splitting text into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,  # Larger chunks for detailed faculty info
        chunk_overlap=100,
        separators=["\n\n", "\n", "- ", "Designation:", "Research Areas:", "Qualification:"]
    )
    split_docs = text_splitter.split_documents(documents)
    print(f"✅ Created {len(split_docs)} text chunks")
    
    # Step 5: Create embeddings
    print("🤖 Loading embedding model...")
    try:
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        print("✅ Embedding model loaded successfully")
    except Exception as e:
        print(f"❌ Error loading embeddings: {e}")
        return False
    
    # Step 6: Create and save vector store
    print("🔗 Creating FAISS vector database...")
    try:
        vectorstore = FAISS.from_documents(split_docs, embeddings)
        
        # Create directory and save
        if not os.path.exists("faculty_vectorstore"):
            os.makedirs("faculty_vectorstore")
        vectorstore.save_local("faculty_vectorstore")
        
        print("✅ Vector database created and saved!")
    except Exception as e:
        print(f"❌ Error creating vector store: {e}")
        return False
    
    # Step 7: Test the retrieval system
    print("\n🧪 Testing retrieval system...")
    try:
        retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
        
        # Test queries
        test_queries = [
            "machine learning professors",
            "deep learning research",
            "computer vision faculty",
            "IoT specialists",
            "cryptography experts",
            "wireless communication",
            "signal processing",
            "data mining",
            "software engineering"
        ]
        
        for query in test_queries:
            results = retriever.get_relevant_documents(query)
            if results:
                print(f"✅ '{query}' → Found {len(results)} relevant results")
            else:
                print(f"⚠️ '{query}' → No results found")
    
    except Exception as e:
        print(f"❌ Error testing retrieval: {e}")
        return False
    
    print(f"\n🎉 RAG System Setup Complete!")
    print(f"📊 System Statistics:")
    print(f"   • Faculty Members: {faculty_count}")
    print(f"   • Text Chunks: {len(split_docs)}")
    print(f"   • Vector Store: Saved to 'faculty_vectorstore/'")
    
    return True

def normalize_research_area(area):
    """Normalize research area names for better categorization"""
    area = area.strip().lower()
    
    # Define mapping for similar terms
    area_mappings = {
        # AI/ML related
        'machine learning': ['ml', 'machine learning', 'artificial intelligence', 'ai'],
        'deep learning': ['deep learning', 'dl', 'neural networks', 'cnn', 'rnn'],
        'computer vision': ['computer vision', 'cv', 'image processing', 'pattern recognition'],
        'natural language processing': ['nlp', 'natural language processing', 'text mining', 'language processing'],
        'data mining': ['data mining', 'data science', 'big data analytics', 'knowledge discovery'],
        
        # Networking/Communication
        'wireless communication': ['wireless', 'wireless communication', 'mobile communication', 'cellular'],
        'network security': ['network security', 'cybersecurity', 'information security'],
        'internet of things': ['iot', 'internet of things', 'sensor networks', 'ubiquitous computing'],
        'signal processing': ['signal processing', 'dsp', 'digital signal processing'],
        
        # Hardware/Systems
        'vlsi design': ['vlsi', 'vlsi design', 'circuit design', 'chip design'],
        'embedded systems': ['embedded systems', 'embedded', 'microcontrollers', 'real time systems'],
        'computer architecture': ['computer architecture', 'processor design', 'system design'],
        
        # Software Engineering
        'software engineering': ['software engineering', 'software development', 'programming'],
        'database systems': ['database', 'dbms', 'data management', 'sql'],
        'algorithms': ['algorithms', 'algorithmic', 'computational complexity'],
        
        # Security/Cryptography
        'cryptography': ['cryptography', 'crypto', 'encryption', 'security protocols'],
        'blockchain': ['blockchain', 'distributed ledger', 'cryptocurrency'],
        
        # Other domains
        'robotics': ['robotics', 'autonomous systems', 'robot control'],
        'optimization': ['optimization', 'mathematical optimization', 'operations research'],
        'bioinformatics': ['bioinformatics', 'computational biology', 'genomics'],
        'game theory': ['game theory', 'algorithmic game theory', 'mechanism design'],
        'quantum computing': ['quantum computing', 'quantum algorithms', 'quantum information'],
    }
    
    # Find matching category
    for category, keywords in area_mappings.items():
        for keyword in keywords:
            if keyword in area:
                return category
    
    # If no match found, return cleaned version of original
    return re.sub(r'[^\w\s]', '', area).strip()

def extract_all_research_areas(content):
    """Extract and categorize all research areas from faculty data"""
    
    # Find all research area lines
    research_lines = []
    lines = content.split('\n')
    
    for line in lines:
        if 'Research Areas:' in line or 'Research Interests:' in line or 'Areas of Interest:' in line:
            # Extract the part after the colon
            if ':' in line:
                areas_text = line.split(':', 1)[1].strip()
                research_lines.append(areas_text)
    
    # Parse all individual research areas
    all_areas = []
    for line in research_lines:
        if line:
            # Split by common delimiters
            areas = re.split(r'[,;]\s*|\s+and\s+|\s*\|\s*', line)
            for area in areas:
                area = area.strip()
                if area and len(area) > 2:  # Filter out very short entries
                    all_areas.append(area)
    
    # Normalize and count
    normalized_areas = Counter()
    area_examples = defaultdict(list)
    
    for area in all_areas:
        normalized = normalize_research_area(area)
        normalized_areas[normalized] += 1
        if len(area_examples[normalized]) < 3:  # Keep up to 3 examples
            area_examples[normalized].append(area)
    
    return normalized_areas, area_examples

def analyze_faculty_data():
    """Comprehensive analysis of the faculty data"""
    
    print("\n📊 COMPREHENSIVE FACULTY DATA ANALYSIS")
    print("=" * 60)
    
    try:
        with open("faculty_details.txt", 'r', encoding='utf-8') as f:
            content = f.read()
    except Exception as e:
        print(f"❌ Error reading file: {e}")
        return
    
    lines = content.split('\n')
    
    # Extract faculty information
    faculty_data = []
    current_faculty = {}
    
    for line in lines:
        line = line.strip()
        if line and line[0].isdigit() and '.' in line:
            # Save previous faculty if exists
            if current_faculty:
                faculty_data.append(current_faculty)
            # Start new faculty
            current_faculty = {'name': line.split('.', 1)[1].strip()}
        elif 'Designation:' in line:
            current_faculty['designation'] = line.replace('Designation:', '').strip()
        elif 'Joint Appointment:' in line:
            current_faculty['department'] = line.replace('Joint Appointment:', '').strip()
        elif any(keyword in line for keyword in ['Research Areas:', 'Research Interests:', 'Areas of Interest:']):
            if ':' in line:
                current_faculty['research_areas'] = line.split(':', 1)[1].strip()
        elif 'Qualification:' in line:
            current_faculty['qualification'] = line.replace('Qualification:', '').strip()
    
    # Don't forget the last faculty
    if current_faculty:
        faculty_data.append(current_faculty)
    
    # Analyze departments
    departments = Counter()
    designations = Counter()
    
    for faculty in faculty_data:
        if 'department' in faculty:
            departments[faculty['department']] += 1
        if 'designation' in faculty:
            designations[faculty['designation']] += 1
    
    # Analyze research areas comprehensively
    research_areas, area_examples = extract_all_research_areas(content)
    
    # Print analysis results
    print(f"👥 FACULTY OVERVIEW:")
    print(f"   Total Faculty: {len(faculty_data)}")
    print(f"   Departments: {len(departments)}")
    print(f"   Unique Research Areas: {len(research_areas)}")
    
    print(f"\n🏢 DEPARTMENTS:")
    for dept, count in departments.most_common():
        print(f"   {dept}: {count} faculty")
    
    print(f"\n👔 DESIGNATIONS:")
    for designation, count in designations.most_common():
        print(f"   {designation}: {count} faculty")
    
    print(f"\n🔬 RESEARCH AREAS (Top 20):")
    for area, count in research_areas.most_common(20):
        examples = ", ".join(area_examples[area][:2])  # Show first 2 examples
        print(f"   {area.title()}: {count} faculty")
        print(f"      Examples: {examples}")
    
    if len(research_areas) > 20:
        print(f"\n   ... and {len(research_areas) - 20} more research areas")
    
    # Show some interesting statistics
    print(f"\n📈 RESEARCH DIVERSITY:")
    total_research_mentions = sum(research_areas.values())
    print(f"   Total Research Area Mentions: {total_research_mentions}")
    print(f"   Average Areas per Faculty: {total_research_mentions / len(faculty_data):.1f}")
    
    # Find most interdisciplinary faculty (those with many research areas)
    faculty_with_most_areas = []
    for faculty in faculty_data:
        if 'research_areas' in faculty:
            area_count = len(re.split(r'[,;]\s*|\s+and\s+|\s*\|\s*', faculty['research_areas']))
            faculty_with_most_areas.append((faculty['name'], area_count))
    
    faculty_with_most_areas.sort(key=lambda x: x[1], reverse=True)
    
    print(f"\n🌟 MOST INTERDISCIPLINARY FACULTY:")
    for name, count in faculty_with_most_areas[:5]:
        print(f"   {name}: {count} research areas")

def generate_sample_queries(research_areas):
    """Generate comprehensive sample queries based on detected research areas"""
    
    queries = []
    
    # Basic area queries
    for area in list(research_areas.keys())[:15]:  # Top 15 areas
        queries.extend([
            f"Who works on {area}?",
            f"Show me {area} researchers",
            f"Which professors do {area} research?",
            f"Tell me about {area} faculty"
        ])
    
    # Combination queries
    area_list = list(research_areas.keys())
    if len(area_list) >= 2:
        queries.extend([
            f"Who works on both {area_list[0]} and {area_list[1]}?",
            f"Faculty doing {area_list[2]} or {area_list[3]} research",
            f"Professors with expertise in {area_list[0]} and related areas"
        ])
    
    # Department-specific queries
    queries.extend([
        "Who is in the CSE department?",
        "Show me all professors",
        "Which faculty have PhD degrees?",
        "Who are the assistant professors?",
        "Tell me about senior faculty members"
    ])
    
    return queries

if __name__ == "__main__":
    print("🎯 LNMIIT Faculty RAG System Builder")
    print("Building RAG system with comprehensive research area analysis\n")
    
    # First analyze the data comprehensively
    analyze_faculty_data()
    
    # Then set up the RAG system
    success = setup_faculty_rag_system()
    
    if success:
        # Generate sample queries based on detected research areas
        try:
            with open("faculty_details.txt", 'r', encoding='utf-8') as f:
                content = f.read()
            research_areas, _ = extract_all_research_areas(content)
            sample_queries = generate_sample_queries(research_areas)
            
            print(f"\n✨ SUCCESS! Your RAG system is ready!")
            print(f"\n📝 Next Steps:")
            print("1. Test the system: python simple_rag_test.py")
            print("2. Or run the full chatbot: python local_rag_app.py")
            print("3. Or try the advanced version: python app.py (needs HuggingFace token)")
            
            print(f"\n💡 Sample queries you can try (based on your data):")
            for i, query in enumerate(sample_queries[:15], 1):  # Show first 15
                print(f"   {i:2d}. '{query}'")
            
            print(f"\n🔍 Advanced query examples:")
            print("   • 'Find experts in both machine learning and computer vision'")
            print("   • 'Who works on IoT and has published recently?'")
            print("   • 'Show me all assistant professors in CSE'")
            print("   • 'Which faculty work on interdisciplinary research?'")
            
        except Exception as e:
            print(f"⚠️ Could not generate custom queries: {e}")
            print(f"\n💡 Try these general queries:")
            print("   • 'Who works on machine learning?'")
            print("   • 'Show me all professors'")
            print("   • 'Which faculty work on AI research?'")
    else:
        print(f"\n💥 Setup failed. Please check the errors above.")
