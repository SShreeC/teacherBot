# 📚 TeacherBot – Your AI-Powered Teaching Assistant

> **An LLM-powered intelligent assistant built for educators to query books, generate lesson plans, prepare questions, and manage classroom activities – all in one AI-enhanced tool.**

---

## 🚀 Overview

**TeacherBot** is a full-featured AI assistant designed specifically for school teachers to simplify academic planning, question creation, and document querying.

Originally built as a personal tool for a teacher (my mom 👩‍🏫), it now includes scalable, class-wise, and multilingual features for broader educator use.

---

## ✨ Features

### 📘 PDF/Textbook Querying
- Upload books or notes (PDFs)
- Ask: _“Give me MCQs from Class 7 Biology Chapter 3”_
- Extracts and processes content to return relevant outputs

### 🗃️ Smart Lesson Plan Generator
- Input: topic, subject, class, and date
- Outputs:
  - Learning objectives
  - Required materials
  - Activities
  - Evaluation criteria

### 🧠 Intelligent Question Generator
- Question types:
  - MCQs
  - Fill in the blanks
  - Long/short answers
  - True/False
- Multilingual support (English, Hindi, more)

### 🏫 Classroom Mode
- Manage class-specific sessions (e.g., Discussion Questions, Concept Explanations, Hands-on Activities, Review Points, Assessment Ideas)
- Stores session history per class
- Customizes tone and content per grade

### 🔁 Session Memory & Follow-Up
- Maintains chat context across multiple queries
- Handles follow-up like _“Now give me 5 more”_

- Save preferences (class, subject, question style, language)

### 🧼 Inappropriate Language Filtering
- Filters profane or off-topic queries
- Keeps app safe for school use

---

## 🧱 Tech Stack

| Layer        | Technologies                            |
|--------------|------------------------------------------|
| 👨‍💻 Backend   |Streamlit(Python)
| 🧠 AI        | Gemini AI, Prompt Engineering            |
| 📄 Docs      | fpdf,pdf2,docx                     |
| 💾 Database  | MySQL                       |


---
### 💻 Local Setup

```bash
git clone https://github.com/SShreeC/teacherbot.git
cd teacherbot
```
### Create a .env file:
```bash PORT=5000
SECRET_KEY
DB_HOST
DB_USER
DB_PASSWORD
```
### Run:
``bash  
streamlit run app.py
```

