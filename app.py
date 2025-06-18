
import streamlit as st
import google.generativeai as genai
from dotenv import load_dotenv
import os
import PyPDF2
import mysql.connector
from fpdf import FPDF
from docx import Document
import logging
import tempfile
import hashlib
from typing import List, Dict, Optional
import re
# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

class Config:
    """Configuration class for application settings"""

    SECRET_KEY = os.getenv("SECRET_KEY")
    DB_HOST = os.getenv("DB_HOST")
    DB_USER = os.getenv("DB_USER")
    DB_PASSWORD = os.getenv("DB_PASSWORD")

    DB_NAME = os.getenv("DB_NAME", "teacherBot")
    MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
    MAX_FILES = 5
    ALLOWED_EXTENSIONS = {'pdf', 'txt'}

# Initialize Gemini AI
if Config.SECRET_KEY:
    genai.configure(api_key=Config.SECRET_KEY)
else:
    st.error("❌ SECRET_KEY not found in environment variables!")
    st.stop()

# Set Streamlit page config
st.set_page_config(
    page_title="📚 AI-Powered Document Q&A",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

class DatabaseManager:
    """Handle database operations securely"""
    
    @staticmethod
    def get_connection():
        """Get database connection with error handling"""
        try:
            conn = mysql.connector.connect(
                host=Config.DB_HOST,
                user=Config.DB_USER,
                password=Config.DB_PASSWORD,
                database=Config.DB_NAME,
                autocommit=True,
                charset='utf8mb4'
            )
            return conn
        except mysql.connector.Error as e:
            logger.error(f"Database connection error: {e}")
            return None
    
    @staticmethod
    def save_document(filename: str, filetype: str, content: str, content_hash: str) -> bool:
        """Save document to database with prepared statements"""
        conn = DatabaseManager.get_connection()
        if not conn:
            return False
        
        try:
            cursor = conn.cursor(prepared=True)
            query = """
                INSERT INTO uploaded_documents (filename, filetype, content, content_hash, upload_time)
                VALUES (?, ?, ?, ?, NOW())
            """
            cursor.execute(query, (filename, filetype, content, content_hash))
            cursor.close()
            conn.close()
            return True
        except mysql.connector.Error as e:
            logger.error(f"Database save error: {e}")
            if conn:
                conn.close()
            return False

class SecurityValidator:
    """Security validation utilities"""
    
    @staticmethod
    def sanitize_filename(filename: str) -> str:
        """Sanitize filename to prevent path traversal"""
        filename = os.path.basename(filename)
        filename = re.sub(r'[^\w\-_\.]', '_', filename)
        return filename[:100]  # Limit length
    
    @staticmethod
    def validate_file_extension(filename: str) -> bool:
        """Validate file extension"""
        ext = filename.lower().split('.')[-1] if '.' in filename else ''
        return ext in Config.ALLOWED_EXTENSIONS
    
    @staticmethod
    def sanitize_content(content: str) -> str:
        """Sanitize content to prevent injection"""
        if not content:
            return ""
        # Remove potential script tags and other dangerous content
        content = re.sub(r'<script[^>]*>.*?</script>', '', content, flags=re.IGNORECASE | re.DOTALL)
        content = re.sub(r'javascript:', '', content, flags=re.IGNORECASE)
        return content[:50000]  # Limit content size

class DocumentProcessor:
    """Handle document processing operations"""
    
    @staticmethod
    def extract_pdf_content(file) -> Optional[str]:
        """Extract text from PDF with error handling"""
        try:
            reader = PyPDF2.PdfReader(file)
            content = []
            for page_num, page in enumerate(reader.pages):
                if page_num > 50:  # Limit pages processed
                    break
                text = page.extract_text()
                if text:
                    content.append(text)
            return "\n".join(content)
        except Exception as e:
            logger.error(f"PDF extraction error: {e}")
            return None
    
    @staticmethod
    def extract_text_content(file) -> Optional[str]:
        """Extract text from text file with error handling"""
        try:
            content = file.read().decode("utf-8")
            return content
        except UnicodeDecodeError:
            try:
                file.seek(0)
                content = file.read().decode("latin-1")
                return content
            except Exception as e:
                logger.error(f"Text extraction error: {e}")
                return None
        except Exception as e:
            logger.error(f"Text file error: {e}")
            return None

class ExportManager:
    """Handle export operations"""
    
    @staticmethod
    def export_to_pdf(text: str, filename: str = "output.pdf") -> Optional[str]:
        """Export text to PDF with error handling"""
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                pdf = FPDF()
                pdf.add_page()
                pdf.set_auto_page_break(auto=True, margin=15)
                pdf.set_font("Arial", size=12)
                
                # Handle text encoding
                text = text.encode('latin-1', 'replace').decode('latin-1')
                
                for line in text.split('\n'):
                    if line.strip():
                        pdf.multi_cell(0, 10, line)
                
                pdf.output(tmp_file.name)
                return tmp_file.name
        except Exception as e:
            logger.error(f"PDF export error: {e}")
            return None
    
    @staticmethod
    def export_to_docx(text: str, filename: str = "output.docx") -> Optional[str]:
        """Export text to DOCX with error handling"""
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.docx') as tmp_file:
                doc = Document()
                for line in text.split('\n'):
                    if line.strip():
                        doc.add_paragraph(line)
                doc.save(tmp_file.name)
                return tmp_file.name
        except Exception as e:
            logger.error(f"DOCX export error: {e}")
            return None

def initialize_session_state():
    """Initialize session state variables"""
    if "documents" not in st.session_state:
        st.session_state.documents = []
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

def validate_uploaded_files(uploaded_files) -> List[Dict]:
    """Validate and process uploaded files"""
    valid_files = []
    
    if not uploaded_files:
        return valid_files
    
    if len(uploaded_files) > Config.MAX_FILES:
        st.warning(f"⚠️ Maximum {Config.MAX_FILES} files allowed. Processing first {Config.MAX_FILES} files.")
        uploaded_files = uploaded_files[:Config.MAX_FILES]
    
    for file in uploaded_files:
        # Validate file size
        if file.size > Config.MAX_FILE_SIZE:
            st.warning(f"⚠️ File {file.name} exceeds maximum size limit (10MB). Skipping.")
            continue
        
        # Sanitize filename
        safe_filename = SecurityValidator.sanitize_filename(file.name)
        
        # Validate file extension
        if not SecurityValidator.validate_file_extension(safe_filename):
            st.warning(f"⚠️ File type not supported for {safe_filename}. Skipping.")
            continue
        
        # Extract content based on file type
        content = None
        if file.type == "application/pdf":
            content = DocumentProcessor.extract_pdf_content(file)
        elif file.type == "text/plain":
            content = DocumentProcessor.extract_text_content(file)
        
        if content:
            # Sanitize content
            safe_content = SecurityValidator.sanitize_content(content)
            if safe_content:
                # Generate content hash for deduplication
                content_hash = hashlib.md5(safe_content.encode()).hexdigest()
                
                valid_files.append({
                    "name": safe_filename,
                    "content": safe_content,
                    "type": file.type,
                    "hash": content_hash
                })
            else:
                st.warning(f"⚠️ No valid content found in {safe_filename}")
        else:
            st.warning(f"⚠️ Failed to extract content from {safe_filename}")
    
    return valid_files

def generate_ai_response(prompt: str, max_retries: int = 3) -> Optional[str]:
    """Generate AI response with error handling and retries"""
    for attempt in range(max_retries):
        try:
            model = genai.GenerativeModel("gemini-2.0-flash")  # Use stable model
            response = model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    max_output_tokens=2048,
                    temperature=0.7,
                )
            )
            return response.text
        except Exception as e:
            logger.error(f"AI generation error (attempt {attempt + 1}): {e}")
            if attempt == max_retries - 1:
                return None
    return None

def main():
    """Main application function"""
    initialize_session_state()
    
    # Sidebar
    st.sidebar.title("⚡ AI Document Assistant")
    st.sidebar.write("Upload documents and ask AI-powered questions!")
    st.sidebar.markdown("---")
    st.sidebar.info("📋 Supported formats: PDF, TXT\n📏 Max file size: 10MB\n📁 Max files: 5")
    
    # Main Title
    st.title("📚 Ask Questions from Uploaded Documents")
    st.write("Upload your documents and get instant answers powered by AI!")
    
    # Upload documents section
    st.subheader("📂 Upload Your Documents")
    uploaded_files = st.file_uploader(
        "Choose PDF/Text files",
        type=list(Config.ALLOWED_EXTENSIONS),
        accept_multiple_files=True,
        help="Supported formats: PDF, TXT (Max 10MB per file, 5 files total)"
    )
    
    temp_save = st.checkbox("💾 Save documents temporarily for session")
    db_save = st.checkbox("🗄️ Save documents to database (permanent storage)")
    
    # Process uploaded files
    if uploaded_files and (temp_save or db_save):
        with st.spinner("Processing uploaded files..."):
            valid_docs = validate_uploaded_files(uploaded_files)
            
            if valid_docs:
                # Save to session state
                st.session_state.documents = valid_docs
                
                # Save to database if requested
                if db_save:
                    saved_count = 0
                    for doc in valid_docs:
                        if DatabaseManager.save_document(
                            doc["name"], doc["type"], doc["content"], doc["hash"]
                        ):
                            saved_count += 1
                    
                    if saved_count > 0:
                        st.success(f"✅ {saved_count} documents saved to database!")
                    else:
                        st.warning("⚠️ Failed to save documents to database")
                
                st.success(f"✅ {len(valid_docs)} documents processed successfully!")
            else:
                st.error("❌ No valid documents were processed")
    
    # Display uploaded documents
    if st.session_state.documents:
        st.subheader("📄 Uploaded Documents")
        for i, doc in enumerate(st.session_state.documents):
            with st.expander(f"📜 {doc['name']} ({len(doc['content'])} characters)"):
                preview_text = doc["content"][:1000]
                if len(doc["content"]) > 1000:
                    preview_text += "..."
                st.text_area("Content Preview", preview_text, height=150, disabled=True, key=f"preview_{i}")
    
    # Question input section
    st.subheader("💬 Ask AI a Question")
    
    # Quick action buttons
    st.markdown("🚀 **Quick Actions**")
    col1, col2, col3, col4 = st.columns(4)
    
    quick_actions = {
        "📄 Summary": "Provide a comprehensive summary of all uploaded documents.",
        "❓ Generate Quiz": "Create a quiz with multiple choice questions based on the documents.",
        "📚 Key Points": "Extract the main key points and important concepts from the documents.",
        "📖 Study Guide": "Create a study guide with important topics and concepts."
    }
    
    selected_action = None
    for i, (label, prompt) in enumerate(quick_actions.items()):
        col = [col1, col2, col3, col4][i]
        if col.button(label, key=f"action_{i}"):
            selected_action = prompt
    
    # Text input for custom questions
    user_input = st.text_input(
        "Type your question here...",
        value=selected_action if selected_action else "",
        help="AI will answer based on the uploaded documents"
    )
    
    # Generate answer
    if st.button("🚀 Get Answer", type="primary"):
        if not st.session_state.documents:
            st.warning("⚠️ Please upload and process documents first!")
        elif not user_input.strip():
            st.warning("⚠️ Please enter a question!")
        else:
            with st.spinner("Generating AI response..."):
                # Combine document content
                combined_content = "\n\n".join([
                    f"Document: {doc['name']}\n{doc['content']}"
                    for doc in st.session_state.documents
                ])
                
                # Create prompt with safety guidelines
                prompt = f"""Based on the following documents, please answer the question accurately and helpfully:

DOCUMENTS:
{combined_content[:10000]}  # Limit context size

QUESTION: {user_input}

Please provide a clear, accurate answer based solely on the information in the provided documents. If the information is not available in the documents, please state that clearly."""
                
                response = generate_ai_response(prompt)
                
                if response:
                    # Add to chat history
                    st.session_state.chat_history.append({
                        "question": user_input,
                        "answer": response
                    })
                    
                    # Display response
                    st.markdown("### 🤖 AI Response:")
                    st.success(response)
                    
                    # Export options for long responses
                    if len(response) > 500:
                        st.markdown("### 📥 Export Options")
                        col1, col2 = st.columns(2)
                        
                        # PDF export
                        pdf_path = ExportManager.export_to_pdf(response)
                        if pdf_path:
                            with open(pdf_path, "rb") as pdf_file:
                                col1.download_button(
                                    "📄 Download as PDF",
                                    pdf_file.read(),
                                    file_name="ai_response.pdf",
                                    mime="application/pdf"
                                )
                        
                        # DOCX export
                        docx_path = ExportManager.export_to_docx(response)
                        if docx_path:
                            with open(docx_path, "rb") as docx_file:
                                col2.download_button(
                                    "📝 Download as DOCX",
                                    docx_file.read(),
                                    file_name="ai_response.docx",
                                    mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
                                )
                else:
                    st.error("❌ Failed to generate AI response. Please try again.")
    
    # Lesson Plan Generator
    st.markdown("---")
    st.subheader("📚 Lesson Plan Generator")
    
    if st.session_state.documents:
        col1, col2 = st.columns([2, 1])
        
        with col1:
            lesson_topic = st.text_input(
                "Lesson Topic (optional)",
                placeholder="e.g., Introduction to Photosynthesis",
                help="Specify a particular topic or leave blank for general lesson plan"
            )
            
            # Language selection for lesson plan
            lesson_language = st.selectbox(
                "🌍 Output Language",
                [
                    "English", "Hindi", "Spanish", "French", "German", "Italian", "Portuguese", 
                    "Russian", "Chinese (Simplified)", "Chinese (Traditional)", "Japanese", 
                    "Korean", "Arabic", "Bengali", "Urdu", "Tamil", "Telugu", "Marathi", 
                    "Gujarati", "Kannada", "Malayalam", "Punjabi", "Dutch", "Swedish", 
                    "Norwegian", "Danish", "Finnish", "Polish", "Czech", "Hungarian", 
                    "Romanian", "Bulgarian", "Greek", "Turkish", "Hebrew", "Thai", 
                    "Vietnamese", "Indonesian", "Malay", "Swahili"
                ],
                index=0,
                help="Select the language for the lesson plan output"
            )
            
            grade_level = st.selectbox(
                "Grade Level",
                ["Elementary (K-5)", "Middle School (6-8)", "High School (9-12)", "College/University", "Adult Education"],
                index=2
            )
            
            lesson_duration = st.selectbox(
                "Lesson Duration",
                ["30 minutes", "45 minutes", "60 minutes", "90 minutes", "2 hours"],
                index=1
            )
        
        with col2:
            st.markdown("**Lesson Plan Components:**")
            st.markdown("✅ Learning Objectives")
            st.markdown("✅ Key Teaching Points")
            st.markdown("✅ Activities & Exercises")
            st.markdown("✅ Assessment Methods")
            st.markdown("✅ Homework/Projects")
            st.markdown("✅ Resources Needed")
        
        if st.button("📋 Generate Comprehensive Lesson Plan", type="primary"):
            with st.spinner("Creating lesson plan..."):
                combined_content = "\n\n".join([
                    f"Document: {doc['name']}\n{doc['content']}"
                    for doc in st.session_state.documents
                ])
                
                lesson_prompt = f"""
                Create a comprehensive lesson plan based on the following document content:
                
                DOCUMENT CONTENT:
                {combined_content[:12000]}
                
                LESSON REQUIREMENTS:
                - Topic: {lesson_topic if lesson_topic else "Based on document content"}
                - Grade Level: {grade_level}
                - Duration: {lesson_duration}
                - OUTPUT LANGUAGE: {lesson_language}
                
                IMPORTANT: Please generate the entire lesson plan in {lesson_language}. All content, headings, explanations, and instructions should be in {lesson_language}.
                
                Please create a detailed lesson plan with the following structure:
                
                1. LESSON OVERVIEW
                   - Title and objectives
                   - Grade level and duration
                   - Materials needed
                
                2. LEARNING OBJECTIVES
                   - What students will learn
                   - Skills they will develop
                
                3. LESSON STRUCTURE
                   - Introduction/Hook (5-10 minutes)
                   - Main content delivery (20-40 minutes)
                   - Activities and practice (15-30 minutes)
                   - Wrap-up and assessment (5-10 minutes)
                
                4. KEY TEACHING POINTS
                   - Main concepts to cover
                   - Important facts and definitions
                
                5. CLASSROOM ACTIVITIES
                   - Interactive exercises
                   - Group work suggestions
                   - Hands-on activities
                
                6. ASSESSMENT METHODS
                   - Formative assessment during class
                   - Summative assessment options
                   - Quiz/test questions
                
                7. HOMEWORK/EXTENSION ACTIVITIES
                   - Take-home assignments
                   - Project ideas
                   - Further reading suggestions
                
                8. DIFFERENTIATION STRATEGIES
                   - For advanced students
                   - For struggling students
                   - For different learning styles
                
                Make it engaging, age-appropriate, and pedagogically sound. Remember to write everything in {lesson_language}.
                """
                
                lesson_response = generate_ai_response(lesson_prompt)
                
                if lesson_response:
                    st.markdown("### 📋 Generated Lesson Plan")
                    st.success(lesson_response)
                    
                    # Add to chat history
                    st.session_state.chat_history.append({
                        "question": f"Generate lesson plan for {lesson_topic or 'document content'} ({grade_level}, {lesson_duration}) in {lesson_language}",
                        "answer": lesson_response
                    })
                    
                    # Export options
                    st.markdown("### 📥 Export Lesson Plan")
                    col1, col2 = st.columns(2)
                    
                    pdf_path = ExportManager.export_to_pdf(lesson_response)
                    if pdf_path:
                        with open(pdf_path, "rb") as pdf_file:
                            col1.download_button(
                                "📄 Download Lesson Plan as PDF",
                                pdf_file.read(),
                                file_name=f"lesson_plan_{lesson_topic.replace(' ', '_') if lesson_topic else 'generated'}.pdf",
                                mime="application/pdf"
                            )
                    
                    docx_path = ExportManager.export_to_docx(lesson_response)
                    if docx_path:
                        with open(docx_path, "rb") as docx_file:
                            col2.download_button(
                                "📝 Download Lesson Plan as DOCX",
                                docx_file.read(),
                                file_name=f"lesson_plan_{lesson_topic.replace(' ', '_') if lesson_topic else 'generated'}.docx",
                                mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
                            )
                else:
                    st.error("❌ Failed to generate lesson plan. Please try again.")
    
    # Classroom Mode
    st.markdown("---")
    st.subheader("🏫 Classroom Mode")
    
    if st.session_state.documents:
        # Document selection
        col1, col2 = st.columns([2, 1])
        
        with col1:
            selected_doc_name = st.selectbox(
                "📄 Choose a document for classroom activities",
                [doc["name"] for doc in st.session_state.documents],
                help="Select which document to use for generating classroom content"
            )
            
            # Language selection for classroom mode
            classroom_language = st.selectbox(
                "🌍 Output Language",
                [
                    "English", "Hindi", "Spanish", "French", "German", "Italian", "Portuguese", 
                    "Russian", "Chinese (Simplified)", "Chinese (Traditional)", "Japanese", 
                    "Korean", "Arabic", "Bengali", "Urdu", "Tamil", "Telugu", "Marathi", 
                    "Gujarati", "Kannada", "Malayalam", "Punjabi", "Dutch", "Swedish", 
                    "Norwegian", "Danish", "Finnish", "Polish", "Czech", "Hungarian", 
                    "Romanian", "Bulgarian", "Greek", "Turkish", "Hebrew", "Thai", 
                    "Vietnamese", "Indonesian", "Malay", "Swahili"
                ],
                index=0,
                help="Select the language for the classroom content output",
                key="classroom_language"
            )
            
            selected_topics = st.text_input(
                "🎯 Specific topics to focus on (optional)",
                placeholder="e.g., photosynthesis, cell division, plant structure",
                help="Enter comma-separated topics or leave blank for general content"
            )
            
            activity_type = st.selectbox(
                "📝 Content Type",
                [
                    "Complete Classroom Package",
                    "Quiz with MCQs",
                    "Discussion Questions",
                    "Hands-on Activities",
                    "Concept Explanations",
                    "Review Materials"
                ]
            )
        
        with col2:
            st.markdown("**Classroom Content Includes:**")
            if activity_type == "Complete Classroom Package":
                st.markdown("✅ Interactive Quiz (5-10 MCQs)")
                st.markdown("✅ Discussion Questions")
                st.markdown("✅ Concept Explanations")
                st.markdown("✅ Hands-on Activities")
                st.markdown("✅ Review Points")
                st.markdown("✅ Assessment Ideas")
            else:
                st.markdown(f"✅ Focused {activity_type}")
                st.markdown("✅ Age-appropriate content")
                st.markdown("✅ Engaging format")
        
        if st.button("🎓 Generate Classroom Content", type="primary"):
            with st.spinner(f"Creating {activity_type.lower()}..."):
                # Get selected document content
                selected_doc = next(
                    doc for doc in st.session_state.documents 
                    if doc["name"] == selected_doc_name
                )
                
                # Create classroom content prompt
                if activity_type == "Complete Classroom Package":
                    classroom_prompt = f"""
                    Based on the following document, create a comprehensive classroom package:
                    
                    DOCUMENT: {selected_doc_name}
                    CONTENT: {selected_doc['content'][:12000]}
                    FOCUS TOPICS: {selected_topics if selected_topics else "All topics in the document"}
                    OUTPUT LANGUAGE: {classroom_language}
                    
                    IMPORTANT: Please generate all content in {classroom_language}. All questions, explanations, activities, and instructions should be in {classroom_language}.
                    
                    Please create:
                    
                    1. INTERACTIVE QUIZ (5-10 Multiple Choice Questions)
                       - Mix of difficulty levels
                       - Clear questions with 4 options each
                       - Correct answers indicated
                       - Brief explanations for answers
                    
                    2. DISCUSSION QUESTIONS (5-7 questions)
                       - Open-ended questions for class discussion
                       - Critical thinking prompts
                       - Questions that encourage analysis
                    
                    3. CONCEPT EXPLANATIONS
                       - Key concepts simplified
                       - Easy-to-understand definitions
                       - Real-world examples and analogies
                    
                    4. HANDS-ON ACTIVITIES (3-5 activities)
                       - Interactive classroom exercises
                       - Group work suggestions
                       - Practical applications
                    
                    5. QUICK REVIEW POINTS
                       - Summary of main ideas
                       - Key facts to remember
                       - Important formulas or concepts
                    
                    6. ASSESSMENT IDEAS
                       - Ways to check understanding
                       - Quick formative assessment suggestions
                    
                    Make it engaging, interactive, and suitable for classroom use. Remember to write everything in {classroom_language}.
                    """
                else:
                    classroom_prompt = f"""
                    Based on the following document, create {activity_type.lower()}:
                    
                    DOCUMENT: {selected_doc_name}
                    CONTENT: {selected_doc['content'][:12000]}
                    FOCUS TOPICS: {selected_topics if selected_topics else "All topics in the document"}
                    OUTPUT LANGUAGE: {classroom_language}
                    
                    IMPORTANT: Please generate all content in {classroom_language}. All text, questions, explanations, and instructions should be in {classroom_language}.
                    
                    Please create detailed {activity_type.lower()} that are:
                    - Engaging and interactive
                    - Age-appropriate
                    - Educational and informative
                    - Ready to use in a classroom setting
                    
                    Format the content clearly with headings and bullet points where appropriate. Remember to write everything in {classroom_language}.
                    """
                
                classroom_response = generate_ai_response(classroom_prompt)
                
                if classroom_response:
                    st.markdown(f"### 🏫 {activity_type}")
                    st.success(classroom_response)
                    
                    # Add to chat history
                    st.session_state.chat_history.append({
                        "question": f"Generate {activity_type} for {selected_doc_name}" + (f" focusing on {selected_topics}" if selected_topics else "") + f" in {classroom_language}",
                        "answer": classroom_response
                    })
                    
                    # Export options
                    st.markdown("### 📥 Export Classroom Content")
                    col1, col2 = st.columns(2)
                    
                    pdf_path = ExportManager.export_to_pdf(classroom_response)
                    if pdf_path:
                        with open(pdf_path, "rb") as pdf_file:
                            col1.download_button(
                                "📄 Download as PDF",
                                pdf_file.read(),
                                file_name=f"classroom_{activity_type.replace(' ', '_').lower()}.pdf",
                                mime="application/pdf"
                            )
                    
                    docx_path = ExportManager.export_to_docx(classroom_response)
                    if docx_path:
                        with open(docx_path, "rb") as docx_file:
                            col2.download_button(
                                "📝 Download as DOCX",
                                docx_file.read(),
                                file_name=f"classroom_{activity_type.replace(' ', '_').lower()}.docx",
                                mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
                            )
                else:
                    st.error("❌ Failed to generate classroom content. Please try again.")
    else:
        st.info("📤 Upload documents first to use Classroom Mode")
    
    # Additional Quick Actions Section
    st.markdown("---")
    st.subheader("⚡ More Quick Actions")
    
    if st.session_state.documents:
        st.markdown("Generate specific content types instantly:")
        
        # Create a grid of quick action buttons
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("📝 Create Study Notes", key="study_notes"):
                quick_prompt = "Create comprehensive study notes with key concepts, definitions, and important points from the uploaded documents. Format them as clear, organized notes suitable for student review."
                st.session_state.quick_action_prompt = quick_prompt
                st.rerun()
            
            if st.button("🧪 Generate Lab/Practical", key="lab_practical"):
                quick_prompt = "Create hands-on laboratory or practical exercises based on the uploaded documents. Include step-by-step procedures, materials needed, and expected outcomes."
                st.session_state.quick_action_prompt = quick_prompt
                st.rerun()
        
        with col2:
            if st.button("📊 Create Presentation Outline", key="presentation"):
                quick_prompt = "Create a detailed presentation outline with main topics, subtopics, and key points from the uploaded documents. Include slide suggestions and talking points."
                st.session_state.quick_action_prompt = quick_prompt
                st.rerun()
            
            if st.button("🎯 Generate Learning Objectives", key="objectives"):
                quick_prompt = "Create clear, measurable learning objectives based on the content in the uploaded documents. Include objectives at different cognitive levels (remember, understand, apply, analyze, evaluate, create)."
                st.session_state.quick_action_prompt = quick_prompt
                st.rerun()
        
        with col3:
            if st.button("📋 Make Checklist/Rubric", key="checklist"):
                quick_prompt = "Create assessment checklists or rubrics based on the uploaded documents. Include criteria for evaluating student understanding and performance."
                st.session_state.quick_action_prompt = quick_prompt
                st.rerun()
            
            if st.button("🔍 Extract Key Terms", key="key_terms"):
                quick_prompt = "Extract and define all important terms, concepts, and vocabulary from the uploaded documents. Create a glossary with clear, concise definitions."
                st.session_state.quick_action_prompt = quick_prompt
                st.rerun()
        
        # Process quick action if one was selected
        if hasattr(st.session_state, 'quick_action_prompt'):
            with st.spinner("Generating content..."):
                combined_content = "\n\n".join([
                    f"Document: {doc['name']}\n{doc['content']}"
                    for doc in st.session_state.documents
                ])
                
                full_prompt = f"""Based on the following documents, {st.session_state.quick_action_prompt}

DOCUMENTS:
{combined_content[:12000]}

Please provide detailed, well-organized content that is immediately useful for educational purposes."""
                
                response = generate_ai_response(full_prompt)
                
                if response:
                    st.markdown("### 🎯 Generated Content")
                    st.success(response)
                    
                    # Add to chat history
                    st.session_state.chat_history.append({
                        "question": st.session_state.quick_action_prompt,
                        "answer": response
                    })
                    
                    # Export options
                    col1, col2 = st.columns(2)
                    pdf_path = ExportManager.export_to_pdf(response)
                    if pdf_path:
                        with open(pdf_path, "rb") as pdf_file:
                            col1.download_button(
                                "📄 Download as PDF",
                                pdf_file.read(),
                                file_name="quick_action_content.pdf",
                                mime="application/pdf"
                            )
                    
                    docx_path = ExportManager.export_to_docx(response)
                    if docx_path:
                        with open(docx_path, "rb") as docx_file:
                            col2.download_button(
                                "📝 Download as DOCX",
                                docx_file.read(),
                                file_name="quick_action_content.docx",
                                mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
                            )
                else:
                    st.error("❌ Failed to generate content. Please try again.")
            
            # Clear the quick action prompt
            del st.session_state.quick_action_prompt
    
    # Chat history
    if st.session_state.chat_history:
        st.subheader("🕘 Chat History")
        
        # Option to clear history
        if st.button("🗑️ Clear Chat History"):
            st.session_state.chat_history = []
            st.rerun()
        
        # Display chat history
        for i, chat in enumerate(reversed(st.session_state.chat_history), 1):
            with st.expander(f"**Q{len(st.session_state.chat_history) - i + 1}: {chat['question']}**"):
                st.info(chat["answer"])
    
    # Footer
    st.markdown("---")
    st.caption("⚠️ Note: Session data will be cleared when you refresh the page.")
    st.markdown("🚀 **Developed with Streamlit & Google Gemini AI**")

if __name__ == "__main__":
    main()