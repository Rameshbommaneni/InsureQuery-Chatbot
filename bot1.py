import gradio as gr
import google.generativeai as genai
import os
import fitz
from sentence_transformers import SentenceTransformer, util
from deep_translator import GoogleTranslator
from collections import deque
import torch

genai.configure(api_key="AIzaSyBov7MBSOeE5mLQw2TX7qAGVWyPMlltSQI")
model = genai.GenerativeModel("gemini-1.5-flash")

PDF_PATHS = [
    r"C:\Users\bomma\OneDrive\Desktop\Insurance Bot\Uploads\Bankers Indemnity Insurance.pdf",
    r"C:\Users\bomma\OneDrive\Desktop\Insurance Bot\Uploads\Bhagyashree Insurance.pdf",
    r"C:\Users\bomma\OneDrive\Desktop\Insurance Bot\Uploads\Burglary Insurance.pdf",
    r"C:\Users\bomma\OneDrive\Desktop\Insurance Bot\Uploads\E Flight Coupon Insurance.pdf",
    r"C:\Users\bomma\OneDrive\Desktop\Insurance Bot\Uploads\Event Cancellation Insurance.pdf",
    r"C:\Users\bomma\OneDrive\Desktop\Insurance Bot\Uploads\Fidelity Guarantee Insurance.pdf",
    r"C:\Users\bomma\OneDrive\Desktop\Insurance Bot\Uploads\group health insurance.pdf",
    r"C:\Users\bomma\OneDrive\Desktop\Insurance Bot\Uploads\House Holder Insurance.pdf",
    r"C:\Users\bomma\OneDrive\Desktop\Insurance Bot\Uploads\Janata Personal Accident Policy.pdf",
    r"C:\Users\bomma\OneDrive\Desktop\Insurance Bot\Uploads\Jewellers Block Insurance.pdf",
    r"C:\Users\bomma\OneDrive\Desktop\Insurance Bot\Uploads\Mahila Udyam Bima.pdf",
    r"C:\Users\bomma\OneDrive\Desktop\Insurance Bot\Uploads\Money Insurance.pdf",
    r"C:\Users\bomma\OneDrive\Desktop\Insurance Bot\Uploads\Neon Sign Insurance.pdf",
    r"C:\Users\bomma\OneDrive\Desktop\Insurance Bot\Uploads\Overseas Mediclaim Insurance Policy(Business&Holiday).pdf",
    r"C:\Users\bomma\OneDrive\Desktop\Insurance Bot\Uploads\Overseas Mediclaim Insureance Policy(Employment&Studies).pdf",
    r"C:\Users\bomma\OneDrive\Desktop\Insurance Bot\Uploads\Package Policy.pdf",
    r"C:\Users\bomma\OneDrive\Desktop\Insurance Bot\Uploads\Plate Glass Insurance.pdf",
    r"C:\Users\bomma\OneDrive\Desktop\Insurance Bot\Uploads\Portable Equipment Insurance.pdf",
    r"C:\Users\bomma\OneDrive\Desktop\Insurance Bot\Uploads\Rasta Apatti Kavach Accident policy.pdf",
    r"C:\Users\bomma\OneDrive\Desktop\Insurance Bot\Uploads\Rasta Apatti Kavach policy.pdf",
    r"C:\Users\bomma\OneDrive\Desktop\Insurance Bot\Uploads\Shopkeeper.pdf"
]

def extract_text_from_pdf(pdf_path):
    try:
        doc = fitz.open(pdf_path)
        text = ""
        for page in doc:
            text += page.get_text()
        doc.close()
        return text
    except Exception as e:
        return f"Error reading {pdf_path}: {str(e)}"

policies = {}
policy_names = []
for pdf_path in PDF_PATHS:
    policy_name = os.path.basename(pdf_path).replace('.pdf', '')
    policy_text = extract_text_from_pdf(pdf_path)
    policies[policy_name] = policy_text
    policy_names.append(policy_name)

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

def get_policy_embeddings():
    return embedding_model.encode(list(policies.values()), convert_to_tensor=True)

policy_embeddings = get_policy_embeddings()

indian_languages = {
    "English (Default)": "en",
    "Telugu": "te",
    "Hindi": "hi",
    "Tamil": "ta",
    "Kannada": "kn",
    "Malayalam": "ml",
    "Marathi": "mr",
    "Bengali": "bn",
    "Gujarati": "gu",
    "Punjabi": "pa",
    "Urdu": "ur",
    "Odia": "or"
}

session_state = {
    "memory": deque(maxlen=10),
    "active_policies": [],
    "active_policies_reason": ""
}

def is_greeting(question):
    greetings = ["hi", "hello", "how are you", "hey", "greetings", "good morning", "good afternoon", "good evening"]
    question_lower = question.lower().strip()
    return any(greeting in question_lower for greeting in greetings)

def is_all_policies_query(question):
    all_policies_keywords = [
        "all policies", "every policy", "all insurance", "list all", "show all",
        "complete list", "all available policies", "entire insurance", "full list"
    ]
    question_lower = question.lower().strip()
    return any(keyword in question_lower for keyword in all_policies_keywords)

def select_top_policies_by_embedding(question, top_k=3):
    question_embedding = embedding_model.encode(question, convert_to_tensor=True)
    cosine_scores = util.cos_sim(question_embedding, policy_embeddings)[0]
    top_results = torch.topk(cosine_scores, k=top_k)
    return [policy_names[i] for i in top_results.indices]

def get_all_policies_summary():
    summary = "Available Insurance Policies:\n\n"
    for policy_name in policy_names:
        summary += f"- {policy_name}: A brief overview of the policy content...\n"
        policy_text = policies[policy_name][:100].strip()
        summary += f"  {policy_text}...\n\n"
    return summary

def ask_ai_with_memory(question, selected_text):
    memory_context = ""
    for q, a in session_state["memory"]:
        if "i am" in q.lower() or "i'm" in q.lower():
            memory_context += f"User Info: {q}\n"
        else:
            memory_context += f"- Q: {q}\n  A: {a}\n"

    full_prompt = f"""
You are an intelligent insurance assistant helping users understand policy documents.

Use the following:
1. User's prior context and memory
2. Only the selected policies relevant to the current question

---
User Context:
{memory_context}

---
Relevant Policy Documents:
{selected_text}

---
Current Question:
{question}

Answer clearly based **only** on the relevant policies and previous conversation context. Avoid guessing or including unrelated policies.
"""
    response = model.generate_content(full_prompt)
    return response.text

def translate_answer(text, target_lang):
    if target_lang == "en":
        return text
    return GoogleTranslator(source='auto', target=target_lang).translate(text)

def insurance_qa(question, language_name, clear_memory=False, explore_other=False):
    global session_state
    selected_language_code = indian_languages[language_name]

    if clear_memory:
        session_state["memory"].clear()
        session_state["active_policies"].clear()
        session_state["active_policies_reason"] = ""
        return "Memory and selected policies cleared.", "", []

    if not question:
        return "Please enter a question.", "", session_state["memory"]

    if is_greeting(question):
        greeting_response = "Hello! I'm here to help with your insurance-related questions. Please ask about any specific policy or insurance topic!"
        translated_response = translate_answer(greeting_response, selected_language_code)
        session_state["memory"].append((question, translated_response))
        return translated_response, "No specific policies selected for greetings.", list(session_state["memory"])

    if is_all_policies_query(question):
        all_policies_text = get_all_policies_summary()
        translated_answer = translate_answer(all_policies_text, selected_language_code)
        session_state["memory"].append((question, translated_answer))
        return translated_answer, "All available policies", list(session_state["memory"])

    if explore_other:
        if not question:
            return "Please enter a new query to explore other policies.", "", session_state["memory"]
        session_state["active_policies"] = select_top_policies_by_embedding(question)
        session_state["active_policies_reason"] = f"(Exploring: {question})"

    if not session_state["active_policies"]:
        top_policies = select_top_policies_by_embedding(question)
        session_state["active_policies"] = top_policies
        session_state["active_policies_reason"] = question
    else:
        top_policies = session_state["active_policies"]

    selected_text = "\n\n".join(policies[name] for name in top_policies)
    answer = ask_ai_with_memory(question, selected_text)
    translated_answer = translate_answer(answer, selected_language_code)
    session_state["memory"].append((question, translated_answer))

    policies_used = f"Using these policies (based on: '{session_state['active_policies_reason']}'):\n- " + "\n- ".join(top_policies)
    return translated_answer, policies_used, list(session_state["memory"])

with gr.Blocks(title="Insurance Q&A Bot") as demo:
    gr.Markdown("# 📄 Insurance Q&A Bot with Memory")
    
    with gr.Row():
        question_input = gr.Textbox(label="💬 Ask your insurance-related question:", placeholder="Enter your question here...")
        language_dropdown = gr.Dropdown(choices=list(indian_languages.keys()), label="🌐 Select Output Language", value="English (Default)")

    with gr.Row():
        clear_button = gr.Button("🧹 Start New Inquiry (Clear Memory)")
        explore_button = gr.Button("🔁 Explore Other Policies")

    output_answer = gr.Textbox(label="🗣️ Answer", lines=10)
    output_policies = gr.Textbox(label="📘 Policies Used", lines=5)
    
    with gr.Accordion("🧠 View Conversation Memory", open=False):
        memory_output = gr.JSON(label="Conversation History")

    question_input.submit(
        fn=insurance_qa,
        inputs=[question_input, language_dropdown, clear_button, explore_button],
        outputs=[output_answer, output_policies, memory_output]
    )

    clear_button.click(
        fn=insurance_qa,
        inputs=[question_input, language_dropdown, gr.State(True), gr.State(False)],
        outputs=[output_answer, output_policies, memory_output]
    )

    explore_button.click(
        fn=insurance_qa,
        inputs=[question_input, language_dropdown, gr.State(False), gr.State(True)],
        outputs=[output_answer, output_policies, memory_output]
    )

demo.launch()