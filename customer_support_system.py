import os
import sys
import json
import sqlite3
import numpy as np
import requests
from typing import TypedDict, List, Dict, Any
from openai import OpenAI
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.sqlite import SqliteSaver

OLLAMA_BASE_URL = "http://localhost:11434/v1"
OLLAMA_EMBED_URL = "http://localhost:11434/api/embeddings"
client = OpenAI(base_url=OLLAMA_BASE_URL, api_key="ollama")

class SupportSystemState(TypedDict):
    customer_id: str
    customer_name: str
    query: str
    category: str
    context: str
    response: str
    requires_approval: bool
    approval_status: str
    final_response: str
    chat_history: List[Dict[str, str]]

class RAGPipeline:
    def __init__(self, doc_dir="knowledge_base"):
        self.doc_dir = doc_dir
        self.chunks = []
        self.embeddings = []
        self.cache_path = os.path.join(self.doc_dir, "embeddings_cache.json")
        self.load_documents()
        self.generate_embeddings()

    def load_documents(self):
        if not os.path.exists(self.doc_dir):
            os.makedirs(self.doc_dir, exist_ok=True)
            return

        for filename in os.listdir(self.doc_dir):
            if filename.endswith(".txt") and not filename.startswith("embeddings_cache"):
                filepath = os.path.join(self.doc_dir, filename)
                with open(filepath, "r", encoding="utf-8") as f:
                    content = f.read()

                sections = content.split("\n\n")
                source_name = filename.replace("_", " ").replace(".txt", "").title()
                for section in sections:
                    clean_section = section.strip()
                    if clean_section:
                        self.chunks.append({
                            "text": f"Source: {source_name}\n\n{clean_section}",
                            "source": source_name
                        })

    def get_embedding(self, text: str) -> List[float]:
        try:
            res = requests.post(OLLAMA_EMBED_URL, json={
                "model": "nomic-embed-text:latest",
                "prompt": text
            })
            res.raise_for_status()
            return res.json()["embedding"]
        except Exception as e:
            return [0.0] * 768

    def generate_embeddings(self):
        if len(self.chunks) == 0:
            return

        if os.path.exists(self.cache_path):
            try:
                with open(self.cache_path, "r", encoding="utf-8") as f:
                    cache_data = json.load(f)
                if len(cache_data) == len(self.chunks):
                    self.embeddings = np.array([item["embedding"] for item in cache_data])
                    return
            except Exception as e:
                pass

        cache_data = []
        for chunk in self.chunks:
            emb = self.get_embedding(chunk["text"])
            self.embeddings.append(emb)
            cache_data.append({"text": chunk["text"], "embedding": emb})
        self.embeddings = np.array(self.embeddings)

        try:
            with open(self.cache_path, "w", encoding="utf-8") as f:
                json.dump(cache_data, f)
        except Exception as e:
            pass

    def retrieve(self, query: str, k: int = 2) -> List[str]:
        if len(self.embeddings) == 0 or len(self.chunks) == 0:
            return ["No documents available in knowledge base."]
        
        query_emb = np.array(self.get_embedding(query))
        dot_product = np.dot(self.embeddings, query_emb)
        norm_docs = np.linalg.norm(self.embeddings, axis=1)
        norm_query = np.linalg.norm(query_emb)
        
        norms = norm_docs * norm_query
        norms[norms == 0] = 1e-9
        
        similarities = dot_product / norms
        top_k_indices = np.argsort(similarities)[::-1][:k]
        
        return [self.chunks[idx]["text"] for idx in top_k_indices]

retriever = RAGPipeline("knowledge_base")

def call_llama(prompt: str, temperature: float = 0.0) -> str:
    try:
        response = client.chat.completions.create(
            model="llama3.2:latest",
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return "I'm sorry, I encountered an internal error. Please try again."

def check_high_risk(query: str, history_str: str) -> bool:
    prompt = f"""
Analyze the customer's query (and conversation history context) and determine if they are requesting any of the following high-risk actions:
1. Refund requests (e.g. asking for money back, refunding subscription)
2. Subscription cancellation (e.g. cancelling the plan, stopping billing)
3. Account closure requests (e.g. deleting the account, closing account, wiping data)
4. Compensation requests (e.g. demanding billing credit, compensation for downtime)
5. Escalation to management (e.g. demanding to speak to a manager, supervisor, or complaining to executives)

Customer Query: "{query}"
History context:
{history_str}

Is the customer requesting one of these actions? Reply with ONLY 'YES' or 'NO'. Do not explain.
"""
    result = call_llama(prompt).upper()
    return "YES" in result

def classify_intent(state: SupportSystemState) -> dict:
    query = state["query"]
    history = state.get("chat_history", [])
    
    history_str = ""
    for msg in history:
        role = "Customer" if msg["role"] == "user" else "Support Agent"
        history_str += f"{role}: {msg['content']}\n"
        
    prompt = f"""
You are an expert customer service router for ABC Technologies.
Your task is to classify the customer's query into one of the following five categories:
1. Sales - Product information, subscription plans, pricing details.
2. Technical - Application errors, installation issues, login problems, configuration issues.
3. Billing - Invoice requests, payment issues, refund requests.
4. Account - Password reset, profile updates, account activation/deactivation.
5. Memory - If the customer is asking about their previous interactions, previous issues, past support history, or what they just said.

Analyze the current query and the chat history (if any).

Chat History:
{history_str}

Current Customer Query:
"{query}"

Output ONLY the category name from this list: [Sales, Technical, Billing, Account, Memory]. Do not write any other text or explanations.
"""
    category = call_llama(prompt)
    category = category.replace("[", "").replace("]", "").strip()
    
    valid_categories = {"Sales", "Technical", "Billing", "Account", "Memory"}
    matched = None
    for cat in valid_categories:
        if cat.lower() in category.lower():
            matched = cat
            break
            
    if matched is None:
        matched = "Technical"
        
    print(f"\n[Node: Classify Intent] Classified query: '{query}' -> Category: {matched}")
    return {"category": matched}

def create_agent_node(department_name: str):
    def agent_node(state: SupportSystemState) -> dict:
        query = state["query"]
        history = state.get("chat_history", [])
        
        context_chunks = retriever.retrieve(query, k=2)
        context_str = "\n\n".join(context_chunks)
        
        history_str = ""
        for msg in history:
            role = "Customer" if msg["role"] == "user" else "Support Agent"
            history_str += f"{role}: {msg['content']}\n"
            
        print(f"[Node: {department_name} Agent] RAG context retrieved. Generating draft response...")
        
        agent_prompt = f"""
You are the specialized {department_name} Support Agent for ABC Technologies.
Your job is to provide accurate, polite, and helpful support to the customer.
Use the following retrieved context documents from the knowledge base to answer the customer's query. If the context doesn't contain the answer, answer as best as you can or guide them appropriately, but do not make up facts.

Retrieved Context:
{context_str}

Chat History:
{history_str}

Customer Query: "{query}"

Formulate a helpful and professional draft response to the customer. Do not mention "Retrieved Context" or "chunks" to the customer. Address them directly.
"""
        draft_response = call_llama(agent_prompt, temperature=0.2)
        requires_approval = check_high_risk(query, history_str)
        approval_status = "pending" if requires_approval else "none"
        
        if requires_approval:
            print(f"[Node: {department_name} Agent] High-risk request detected! Flagging for human approval.")
            
        return {
            "context": context_str,
            "response": draft_response,
            "requires_approval": requires_approval,
            "approval_status": approval_status
        }
    return agent_node

sales_agent = create_agent_node("Sales")
technical_agent = create_agent_node("Technical Support")
billing_agent = create_agent_node("Billing")
account_agent = create_agent_node("Account")

def memory_recall_agent(state: SupportSystemState) -> dict:
    history = state.get("chat_history", [])
    history_str = ""
    for msg in history:
        role = "Customer" if msg["role"] == "user" else "Support Agent"
        history_str += f"{role}: {msg['content']}\n"
        
    print("[Node: Memory Agent] Retrieving conversation history from SQLite memory...")
    
    prompt = f"""
You are the Memory Recall Support Agent for ABC Technologies.
The customer is asking what they requested or asked about previously.

Here is the conversation history:
{history_str}

Based on the conversation history above, identify and summarize what the customer requested or asked about, and what the response was.
- If there are messages in the history, summarize them directly. Do NOT say this is the customer's first interaction.
- If the history is empty, say: "This is your first interaction with us, so there is no previous history."
Keep your response concise and professional.
"""
    response = call_llama(prompt, temperature=0.1)
    
    return {
        "context": "Retrieved from conversation memory database.",
        "response": response,
        "requires_approval": False,
        "approval_status": "none"
    }

def supervisor_agent(state: SupportSystemState) -> dict:
    draft = state["response"]
    context = state.get("context", "")
    query = state["query"]
    
    print("[Node: Supervisor] Reviewing and validating draft response...")
    
    prompt = f"""
You are the QA Supervisor Agent for ABC Technologies.
Your job is to validate and improve the draft response generated by the support agent.
Ensure that:
1. The response is factually accurate according to the Retrieved Context.
2. The response is polite, empathetic, and professional.
3. If the request is high-risk (e.g. refund, cancellation, closure, compensation, escalation), the response must state that the request has been submitted to a human supervisor for review and approval, and must NOT promise immediate approval.

Retrieved Context:
{context}

Customer Query: "{query}"
Draft Agent Response: "{draft}"

Generate a polished and improved version of this response. Follow all the guidelines. Keep the tone professional.
If the draft response is already perfect, output it exactly. Output ONLY the finalized response text. Do not add any preamble, supervisor notes, or metadata.
"""
    improved_response = call_llama(prompt, temperature=0.1)
    return {
        "response": improved_response
    }

def human_approval(state: SupportSystemState) -> dict:
    status = state.get("approval_status", "pending")
    draft = state.get("response", "")
    
    print(f"[Node: Human Approval] Running approval gate. Current status: {status}")
    
    if status == "approved":
        final = draft
    elif status == "rejected":
        final = "Your request has been reviewed by a supervisor and could not be approved at this time. An escalation ticket has been created, and our team will contact you shortly."
    else:
        final = draft
        
    history = list(state.get("chat_history", []))
    history.append({"role": "user", "content": state["query"]})
    history.append({"role": "assistant", "content": final})
    
    return {
        "final_response": final,
        "chat_history": history,
        "approval_status": status
    }

def finalize_response(state: SupportSystemState) -> dict:
    print("[Node: Finalize Response] Finalizing and saving response to history...")
    draft = state["response"]
    
    history = list(state.get("chat_history", []))
    history.append({"role": "user", "content": state["query"]})
    history.append({"role": "assistant", "content": draft})
    
    return {
        "final_response": draft,
        "chat_history": history
    }

def route_after_classify(state: SupportSystemState) -> str:
    category = state["category"]
    if category == "Memory":
        return "memory"
    elif category == "Sales":
        return "sales"
    elif category == "Technical":
        return "technical"
    elif category == "Billing":
        return "billing"
    elif category == "Account":
        return "account"
    else:
        return "technical"

def route_after_supervisor(state: SupportSystemState) -> str:
    if state.get("requires_approval") and state.get("approval_status") == "pending":
        print("[Router] Query requires human approval. Directing to human approval node.")
        return "human_approval"
    else:
        print("[Router] Query does not require approval or is already approved/rejected. Directing to finalize.")
        return "finalize"

db_path = "support_memory.db"
conn = sqlite3.connect(db_path, check_same_thread=False)
checkpointer = SqliteSaver(conn)

workflow = StateGraph(SupportSystemState)

workflow.add_node("classify", classify_intent)
workflow.add_node("sales_agent", sales_agent)
workflow.add_node("technical_agent", technical_agent)
workflow.add_node("billing_agent", billing_agent)
workflow.add_node("account_agent", account_agent)
workflow.add_node("memory_agent", memory_recall_agent)
workflow.add_node("supervisor_agent", supervisor_agent)
workflow.add_node("human_approval", human_approval)
workflow.add_node("finalize_response", finalize_response)

workflow.set_entry_point("classify")

workflow.add_conditional_edges(
    "classify",
    route_after_classify,
    {
        "sales": "sales_agent",
        "technical": "technical_agent",
        "billing": "billing_agent",
        "account": "account_agent",
        "memory": "memory_agent"
    }
)

workflow.add_edge("sales_agent", "supervisor_agent")
workflow.add_edge("technical_agent", "supervisor_agent")
workflow.add_edge("billing_agent", "supervisor_agent")
workflow.add_edge("account_agent", "supervisor_agent")
workflow.add_edge("memory_agent", "supervisor_agent")

workflow.add_conditional_edges(
    "supervisor_agent",
    route_after_supervisor,
    {
        "human_approval": "human_approval",
        "finalize": "finalize_response"
    }
)

workflow.add_edge("human_approval", END)
workflow.add_edge("finalize_response", END)

graph = workflow.compile(
    checkpointer=checkpointer,
    interrupt_before=["human_approval"]
)

def run_query(query: str, thread_id: str, customer_name: str, auto_approve: bool = True):
    print("\n" + "="*80)
    print(f"USER QUERY: \"{query}\"")
    print(f"Thread ID: {thread_id} | Customer: {customer_name}")
    print("="*80)
    
    config = {"configurable": {"thread_id": thread_id}}
    
    inputs = {
        "customer_id": thread_id,
        "customer_name": customer_name,
        "query": query,
    }
    
    print("[*] Starting LangGraph execution...")
    state_output = graph.invoke(inputs, config)
    
    state_info = graph.get_state(config)
    
    if "human_approval" in state_info.next:
        current_state = state_info.values
        print("\n" + "#"*40)
        print("!!! HUMAN INTERRUPT DETECTED !!!")
        print("#"*40)
        print(f"Query: \"{current_state['query']}\"")
        print(f"Proposed Response (QA validated): \n{current_state['response']}")
        print("-"*40)
        
        if auto_approve:
            print("[Simulated Human Supervisor]: Reviewing request... refund/cancellation detected. APPROVED.")
            approval_decision = "approved"
        else:
            sys.stdout.flush()
            decision = input("Review Response. Approve? (Y/N/Edit): ").strip().lower()
            if decision == 'y':
                approval_decision = "approved"
            elif decision == 'n':
                approval_decision = "rejected"
            else:
                approval_decision = "approved"
                
        print(f"[*] Updating state: approval_status -> {approval_decision}")
        graph.update_state(config, {"approval_status": approval_decision})
        
        print("[*] Resuming LangGraph execution...")
        state_output = graph.invoke(None, config)
    
    final_state = graph.get_state(config).values
    print("\n" + "="*80)
    print("FINAL RESPONSE SENT TO CUSTOMER:")
    print("="*80)
    print(final_state.get("final_response"))
    print("="*80)
    
    print(f"[*] Total history length in thread '{thread_id}': {len(final_state.get('chat_history', []))} messages.")
    
    return final_state

def main():
    print("[+] ABC Technologies Support System CLI.")
    customer_name = input("Enter your name: ").strip()
    if not customer_name:
        customer_name = "Guest"
        
    thread_id = f"thread_{customer_name.lower().replace(' ', '_')}"
    print(f"[+] Session started for {customer_name} (Thread ID: {thread_id}). Type 'exit' to quit.")
        
    while True:
        try:
            user_input = input(f"\n{customer_name} > ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye!")
            break
                
        if not user_input:
            continue
                
        if user_input.lower() in ["exit", "quit", "bye"]:
            print("Goodbye!")
            break
                
        run_query(user_input, thread_id, customer_name, auto_approve=True)

if __name__ == "__main__":
    main()
