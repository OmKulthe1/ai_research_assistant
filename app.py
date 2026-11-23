import streamlit as st
import os
import sys
import json
from dotenv import load_dotenv

# Ensure local module access
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.agent import AutonomousResearchAgent, _extract_text
from src.memory import LongTermMemory, ShortTermMemory
from src.tools.rag import filter_top_k_sources
from src.tools.pdf_reader import process_pdf_files

# Page configuration
st.set_page_config(
    page_title="Autonomous AI Research Assistant",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern multi-agent design
st.markdown("""
<style>
    /* Global styles */
    .main {
        background-color: #0d1117;
        color: #c9d1d9;
    }
    .stApp {
        background: linear-gradient(135deg, #0d1117 0%, #161b22 100%);
    }
    /* Card container styling */
    .metric-card {
        background: rgba(22, 27, 34, 0.85);
        border: 1px solid #30363d;
        border-radius: 10px;
        padding: 14px 18px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
        margin-bottom: 12px;
    }
    /* Agent role card */
    .agent-card {
        background: rgba(22, 27, 34, 0.9);
        border: 1px solid #30363d;
        border-radius: 8px;
        padding: 12px;
        text-align: center;
        transition: transform 0.2s, border-color 0.2s;
    }
    .agent-card:hover {
        border-color: #58a6ff;
        transform: translateY(-2px);
    }
    /* Header title gradient */
    .gradient-header {
        background: linear-gradient(90deg, #79c0ff 0%, #d2a8ff 50%, #ffa657 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
        font-size: 2.3rem;
        margin-bottom: 0.2rem;
    }
    /* Subtitle */
    .subtitle {
        color: #8b949e;
        font-size: 1.05rem;
        margin-bottom: 1.6rem;
    }
    /* Source pill badge */
    .badge-arxiv {
        background-color: #1f6feb;
        color: #ffffff;
        padding: 3px 8px;
        border-radius: 12px;
        font-size: 0.75rem;
        font-weight: 600;
    }
    .badge-web {
        background-color: #238636;
        color: #ffffff;
        padding: 3px 8px;
        border-radius: 12px;
        font-size: 0.75rem;
        font-weight: 600;
    }
    .badge-memory {
        background-color: #8957e5;
        color: #ffffff;
        padding: 3px 8px;
        border-radius: 12px;
        font-size: 0.75rem;
        font-weight: 600;
    }
    .badge-pdf {
        background-color: #bf4b8a;
        color: #ffffff;
        padding: 3px 8px;
        border-radius: 12px;
        font-size: 0.75rem;
        font-weight: 600;
    }
    /* Performative badges */
    .badge-cfp {
        background-color: #a371f7;
        color: #ffffff;
        padding: 2px 7px;
        border-radius: 10px;
        font-size: 0.7rem;
        font-weight: bold;
    }
    .badge-propose {
        background-color: #388bfd;
        color: #ffffff;
        padding: 2px 7px;
        border-radius: 10px;
        font-size: 0.7rem;
        font-weight: bold;
    }
    .badge-accept {
        background-color: #2ea043;
        color: #ffffff;
        padding: 2px 7px;
        border-radius: 10px;
        font-size: 0.7rem;
        font-weight: bold;
    }
    .badge-inform {
        background-color: #39c5cf;
        color: #000000;
        padding: 2px 7px;
        border-radius: 10px;
        font-size: 0.7rem;
        font-weight: bold;
    }
    .badge-request {
        background-color: #d29922;
        color: #000000;
        padding: 2px 7px;
        border-radius: 10px;
        font-size: 0.7rem;
        font-weight: bold;
    }
    /* History card */
    .history-card {
        background: rgba(22, 27, 34, 0.95);
        border: 1px solid #30363d;
        border-radius: 8px;
        padding: 10px;
        margin-bottom: 8px;
        transition: border-color 0.2s;
    }
    .history-card:hover {
        border-color: #58a6ff;
    }
</style>
""", unsafe_allow_html=True)

# Load environment
load_dotenv()

# Initialize Persistent Long-Term Memory Store
long_term_mem = LongTermMemory()

# Session State Initialization
if "current_session_id" not in st.session_state:
    st.session_state.current_session_id = None
if "active_research" not in st.session_state:
    st.session_state.active_research = None
if "pending_proposal" not in st.session_state:
    st.session_state.pending_proposal = None
if "proposal_agent" not in st.session_state:
    st.session_state.proposal_agent = None

# Header Section
st.markdown('<div class="gradient-header">🧠 Autonomous AI Research Assistant</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Role-Based Multi-Agent Collaboration & Contract Net Protocol (CNP) Orchestration</div>', unsafe_allow_html=True)

# ==========================================
# SIDEBAR CONTROLS & LONG-TERM MEMORY
# ==========================================
with st.sidebar:
    st.image("https://img.icons8.com/color/96/brain.png", width=56)
    st.header("⚙️ Agent Controls")
    
    api_key_env = os.getenv("GOOGLE_API_KEY", "")
    has_api_key = bool(api_key_env and api_key_env != "your_gemini_api_key_here")
    
    if not has_api_key:
        user_api_key = st.text_input("🔑 Enter Gemini API Key", type="password", help="Get free key from https://aistudio.google.com/")
        if user_api_key:
            os.environ["GOOGLE_API_KEY"] = user_api_key
            st.success("API Key applied for this session!")
            has_api_key = True
        else:
            st.warning("⚠️ API Key required for online Gemini models (or select offline Ollama/Qwen below).")
    else:
        st.success("🟢 API Key Configured (.env)")
        
    model_choice = st.selectbox(
        "🧠 LLM Engine",
        [
            "gemini-3.6-flash",
            "gemini-3.5-flash",
            "gemini-3.7-flash",
            "gemini-flash-latest",
            "ollama/qwen2.5:3b",
            "ollama/qwen:3b",
            "ollama/llama3.2:3b"
        ],
        index=0,
        help="Select Google Gemini API (Online) or Ollama/Qwen (Offline Local)"
    )

    st.markdown("### 🎛️ Autonomy Level")
    autonomy_mode = st.radio(
        "Execution Mode:",
        [
            "🤖 Full Autonomy (End-to-End)",
            "🤝 Supervised (Human-in-the-Loop)"
        ],
        index=0,
        help="Full Autonomy executes the full cycle automatically. Supervised pauses after bidding so you can review and edit queries before search execution."
    )

    st.divider()

    st.markdown("### 📄 Deep Document Ingestion")
    enable_deep_arxiv = st.checkbox(
        "🔬 Deep arXiv PDF Reader",
        value=True,
        help="Autonomously downloads and reads full-text arXiv PDFs for detailed methodologies & tables."
    )
    uploaded_pdf_files = st.file_uploader(
        "📂 Upload Research PDFs",
        type=["pdf"],
        accept_multiple_files=True,
        help="Upload custom research papers or reports to be ingested alongside arXiv and Web sources."
    )
    parsed_custom_sources = []
    if uploaded_pdf_files:
        parsed_custom_sources = process_pdf_files(uploaded_pdf_files)
        st.success(f"Ingested {len(parsed_custom_sources)} section(s) from {len(uploaded_pdf_files)} PDF(s).")

    st.divider()

    # Long-Term Memory Section
    st.markdown("### 🗄️ Long-Term Memory Store")
    st.caption("Persistent SQLite storage across research sessions")
    
    if st.button("➕ Start New Research Topic", use_container_width=True, type="secondary"):
        st.session_state.current_session_id = None
        st.session_state.active_research = None
        st.session_state.pending_proposal = None
        st.session_state.proposal_agent = None
        st.rerun()

    # Search past sessions
    search_mem = st.text_input("🔍 Search Past Research", placeholder="e.g. quantum, game...", key="search_mem_box")
    if search_mem.strip():
        past_sessions = long_term_mem.search_sessions(search_mem.strip())
    else:
        past_sessions = long_term_mem.list_sessions(limit=10)

    if past_sessions:
        st.markdown(f"**Saved Dossiers ({len(past_sessions)}):**")
        for sess in past_sessions:
            sid = sess["session_id"]
            is_active = (sid == st.session_state.current_session_id)
            title = sess.get("topic_title") or sess.get("objective", "Untitled")[:40]
            created = sess.get("created_at", "")[:10]
            
            with st.container():
                st.markdown(f"""
                <div class="history-card" style="border-left: 3px solid {'#7ee787' if is_active else '#30363d'};">
                    <strong style="color: {'#7ee787' if is_active else '#c9d1d9'}; font-size: 0.85rem;">{title}</strong><br>
                    <span style="color: #8b949e; font-size: 0.75rem;">📅 {created} | ID: <code>{sid}</code></span>
                </div>
                """, unsafe_allow_html=True)
                
                c_load, c_del = st.columns([3, 1])
                with c_load:
                    if st.button("📂 Load", key=f"load_{sid}", use_container_width=True):
                        loaded_data = long_term_mem.get_session(sid)
                        if loaded_data:
                            st.session_state.current_session_id = sid
                            st.session_state.active_research = loaded_data
                            st.session_state.pending_proposal = None
                            st.session_state.proposal_agent = None
                            st.rerun()
                with c_del:
                    if st.button("🗑️", key=f"del_{sid}", use_container_width=True):
                        long_term_mem.delete_session(sid)
                        if st.session_state.current_session_id == sid:
                            st.session_state.current_session_id = None
                            st.session_state.active_research = None
                        st.rerun()
    else:
        st.caption("No past research sessions found. Run an objective to persist.")

    st.divider()
    st.markdown("### 📌 Demo Presets")
    selected_preset = st.radio("Select Demo Topic:", [
        "🎮 1. Game Design & Game Development",
        "🤖 2. Autonomous AI Agents",
        "⚛️ 3. Quantum Computing"
    ])

    if selected_preset.startswith("🎮"):
        default_obj = "Research the latest developments in Game Development for game design and prepare a summarized report."
    elif selected_preset.startswith("🤖"):
        default_obj = "Research recent developments in Autonomous AI Agents and Multi-Agent Systems."
    else:
        default_obj = "Research the latest developments in Quantum Computing and prepare a summarized report."

# ==========================================
# MULTI-AGENT TEAM OVERVIEW
# ==========================================
st.markdown("### 👥 Role-Based Multi-Agent Team")
c_ag1, c_ag2, c_ag3, c_ag4 = st.columns(4)

with c_ag1:
    st.markdown("""
    <div class="agent-card">
        <div style="font-size: 1.6rem;">👔</div>
        <strong style="color: #79c0ff; font-size: 0.9rem;">Research Director</strong><br>
        <span style="color: #8b949e; font-size: 0.75rem;">Role: <b>Coordinator</b><br>CNP Manager & Synthesis</span>
    </div>
    """, unsafe_allow_html=True)

with c_ag2:
    st.markdown("""
    <div class="agent-card">
        <div style="font-size: 1.6rem;">🎓</div>
        <strong style="color: #7ee787; font-size: 0.9rem;">Academic Specialist</strong><br>
        <span style="color: #8b949e; font-size: 0.75rem;">Role: <b>Contractor</b><br>arXiv Literature & Preprints</span>
    </div>
    """, unsafe_allow_html=True)

with c_ag3:
    st.markdown("""
    <div class="agent-card">
        <div style="font-size: 1.6rem;">🌐</div>
        <strong style="color: #ffa657; font-size: 0.9rem;">Industry Scout</strong><br>
        <span style="color: #8b949e; font-size: 0.75rem;">Role: <b>Contractor</b><br>Live Web & Tech News</span>
    </div>
    """, unsafe_allow_html=True)

with c_ag4:
    st.markdown("""
    <div class="agent-card">
        <div style="font-size: 1.6rem;">🔬</div>
        <strong style="color: #d2a8ff; font-size: 0.9rem;">Quality Auditor</strong><br>
        <span style="color: #8b949e; font-size: 0.75rem;">Role: <b>Monitor</b><br>Peer Review & Grounding</span>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ==========================================
# MAIN INTERACTION AREA
# ==========================================

# Current research objective input
current_obj = ""
if st.session_state.active_research:
    current_obj = st.session_state.active_research.get("objective", "")

objective_input = st.text_area(
    "🎯 Research Objective (enter your domain or question):",
    value=current_obj if current_obj else default_obj,
    height=85
)

col_run, col_status = st.columns([1, 3])
with col_run:
    if autonomy_mode.startswith("🤖"):
        start_btn = st.button("🚀 Run Full Autonomy", use_container_width=True, type="primary")
    else:
        start_btn = st.button("📋 Generate Plan & Solicit Bids", use_container_width=True, type="primary")

# Execute Multi-Agent Pipeline or Solicit Bids for HITL
if start_btn:
    is_offline = "ollama" in model_choice.lower()
    if not is_offline and (not os.getenv("GOOGLE_API_KEY") or os.getenv("GOOGLE_API_KEY") == "your_gemini_api_key_here"):
        st.error("❌ Please provide a valid GOOGLE_API_KEY in the sidebar or `.env` file for online inference.")
    else:
        if autonomy_mode.startswith("🤖"):
            # Full Autonomy Mode: End-to-End Execution
            status_box = st.status("🔄 Multi-Agent Team Collaborating in Full Autonomy Mode...", expanded=True)
            try:
                agent = AutonomousResearchAgent(model_name=model_choice)
                
                def _ui_callback(msg, step):
                    status_box.write(f"⚙️ **Phase {step}:** {msg}")

                plan, filtered_sources, draft_report, final_report = agent.run_pipeline(
                    objective=objective_input,
                    progress_callback=_ui_callback,
                    max_reflection_loops=2,
                    deep_read=enable_deep_arxiv,
                    custom_sources=parsed_custom_sources
                )

                cnp_logs = [m.dict() for m in agent.cnp_message_log]
                reflection_logs = list(agent.reflection_history)
                session_id = agent.current_session_id or f"sess_{os.urandom(4).hex()}"
                topic_title = objective_input[:60] + ("..." if len(objective_input) > 60 else "")

                st.session_state.current_session_id = session_id
                st.session_state.active_research = {
                    "session_id": session_id,
                    "objective": objective_input,
                    "topic_title": topic_title,
                    "model_name": model_choice,
                    "plan": plan.dict() if hasattr(plan, "dict") else plan,
                    "sources": filtered_sources,
                    "final_report": final_report,
                    "chat_history": [],
                    "cnp_messages": cnp_logs,
                    "reflection_history": reflection_logs
                }
                st.session_state.pending_proposal = None
                st.session_state.proposal_agent = None

                status_box.update(label="✅ Multi-Agent Collaboration & Self-Correction Complete!", state="complete")
                st.rerun()

            except Exception as e:
                status_box.update(label="❌ Multi-Agent Execution Error", state="error")
                st.error(f"Error during agent pipeline: {e}")

        else:
            # Supervised (HITL) Mode: Solicit Bids & Pause at Approval Gate
            status_box = st.status("📋 Soliciting Contractor Bids for Human Review...", expanded=True)
            try:
                agent = AutonomousResearchAgent(model_name=model_choice)
                proposal_data = agent.prepare_plan_for_approval(objective_input)
                
                st.session_state.pending_proposal = proposal_data
                st.session_state.proposal_agent = agent
                st.session_state.active_research = None
                status_box.update(label="⏸️ Bids Received! Paused at Human-in-the-Loop Approval Gate.", state="complete")
                st.rerun()
            except Exception as e:
                status_box.update(label="❌ Planning Error", state="error")
                st.error(f"Error during bid solicitation: {e}")

# ==========================================
# HUMAN-IN-THE-LOOP APPROVAL GATE (HITL)
# ==========================================
if st.session_state.pending_proposal:
    prop = st.session_state.pending_proposal
    st.markdown("""
    <div style="background: rgba(22, 27, 34, 0.95); border: 2px solid #58a6ff; border-radius: 10px; padding: 18px; margin-top: 15px; margin-bottom: 20px;">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px;">
            <div>
                <h3 style="margin: 0; color: #58a6ff;">🛡️ Human-in-the-Loop Approval Gate</h3>
                <span style="color: #8b949e; font-size: 0.85rem;">Review, refine, or authorize the agents' query bids before search contracts execute.</span>
            </div>
            <span style="background: #1f6feb; color: #ffffff; padding: 4px 12px; border-radius: 12px; font-size: 0.75rem; font-weight: bold;">PAUSED FOR AUTHORIZATION</span>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("**🎯 Decomposed Sub-Questions:**")
    for sq in prop.get("sub_questions", []):
        st.markdown(f"- {sq}")

    c_q1, c_q2 = st.columns(2)
    with c_q1:
        st.markdown(f"**🎓 Academic Specialist (arXiv Bids)** *(Confidence: {int(prop.get('acad_confidence', 0.9)*100)}%)*:")
        st.caption(f"_{prop.get('acad_rationale', '')}_")
        acad_text = st.text_area(
            "arXiv Search Vectors (edit queries below, one per line):",
            value="\n".join(prop.get("arxiv_queries", [])),
            height=90,
            key="hitl_arxiv_input"
        )
    with c_q2:
        st.markdown(f"**🌐 Industry Scout (Web Bids)** *(Confidence: {int(prop.get('ind_confidence', 0.9)*100)}%)*:")
        st.caption(f"_{prop.get('ind_rationale', '')}_")
        web_text = st.text_area(
            "Web / Industry Search Vectors (edit queries below, one per line):",
            value="\n".join(prop.get("web_queries", [])),
            height=90,
            key="hitl_web_input"
        )

    c_auth, c_cancel, _ = st.columns([2, 1, 3])
    with c_auth:
        authorize_btn = st.button("✅ Authorize & Execute Contracts", type="primary", use_container_width=True)
    with c_cancel:
        cancel_btn = st.button("❌ Cancel Proposal", use_container_width=True)

    st.markdown("</div>", unsafe_allow_html=True)

    if cancel_btn:
        st.session_state.pending_proposal = None
        st.session_state.proposal_agent = None
        st.rerun()

    if authorize_btn:
        edited_arxiv = [q.strip() for q in acad_text.split("\n") if q.strip()]
        edited_web = [q.strip() for q in web_text.split("\n") if q.strip()]

        from src.schemas import ResearchPlan
        approved_plan = ResearchPlan(
            sub_questions=prop.get("sub_questions", []),
            arxiv_queries=edited_arxiv if edited_arxiv else prop.get("arxiv_queries", []),
            web_queries=edited_web if edited_web else prop.get("web_queries", [])
        )

        agent = st.session_state.proposal_agent or AutonomousResearchAgent(model_name=model_choice)
        status_box = st.status("🔄 Executing Authorized Multi-Agent Research...", expanded=True)

        try:
            def _ui_callback(msg, step):
                status_box.write(f"⚙️ **Phase {step}:** {msg}")

            plan, filtered_sources, draft_report, final_report = agent.execute_from_approved_plan(
                objective=prop.get("objective", objective_input),
                approved_plan=approved_plan,
                progress_callback=_ui_callback,
                max_reflection_loops=2,
                deep_read=enable_deep_arxiv,
                custom_sources=parsed_custom_sources
            )

            cnp_logs = [m.dict() for m in agent.cnp_message_log]
            reflection_logs = list(agent.reflection_history)
            session_id = agent.current_session_id or f"sess_{os.urandom(4).hex()}"
            topic_title = prop.get("objective", objective_input)[:60] + ("..." if len(prop.get("objective", objective_input)) > 60 else "")

            st.session_state.current_session_id = session_id
            st.session_state.active_research = {
                "session_id": session_id,
                "objective": prop.get("objective", objective_input),
                "topic_title": topic_title,
                "model_name": model_choice,
                "plan": plan.dict() if hasattr(plan, "dict") else plan,
                "sources": filtered_sources,
                "final_report": final_report,
                "chat_history": [],
                "cnp_messages": cnp_logs,
                "reflection_history": reflection_logs
            }
            st.session_state.pending_proposal = None
            st.session_state.proposal_agent = None

            status_box.update(label="✅ Authorized Multi-Agent Research Complete!", state="complete")
            st.rerun()
        except Exception as e:
            status_box.update(label="❌ Multi-Agent Execution Error", state="error")
            st.error(f"Error during authorized execution: {e}")

# ==========================================
# RENDER ACTIVE RESEARCH DOSSIER
# ==========================================
if st.session_state.active_research:
    active = st.session_state.active_research
    sid = active.get("session_id", "N/A")
    plan_data = active.get("plan", {})
    sources = active.get("sources", [])
    final_report = active.get("final_report", "")
    chat_history = active.get("chat_history", [])
    cnp_messages = active.get("cnp_messages", [])

    st.markdown(f"""
    <div class="metric-card" style="display: flex; justify-content: space-between; align-items: center;">
        <div>
            <span class="badge-memory">Session ID: {sid}</span>&nbsp;
            <span style="color: #79c0ff; font-weight: 600;">Active Topic:</span> {active.get('objective', '')[:80]}
        </div>
        <div style="color: #8b949e; font-size: 0.8rem;">
            Model: <code>{active.get('model_name', model_choice)}</code>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Autonomous Self-Correction & Reflection Timeline
    reflection_history = active.get("reflection_history", [])
    if reflection_history:
        with st.expander(f"🔄 Autonomous Self-Correction & Reflection Timeline ({len(reflection_history)} iteration(s))", expanded=True):
            st.caption("Iterative peer-review reflection loop: Quality Auditor audits factual grounding and directs remedial search until quality standards are met.")
            c_cols = st.columns(max(len(reflection_history), 1))
            for i, rec in enumerate(reflection_history):
                with c_cols[i]:
                    it_num = rec.get("iteration", i + 1)
                    g_score = rec.get("grounding_score", 0)
                    d_score = rec.get("citation_density_score", 0)
                    status_text = rec.get("status", "Approved")
                    badge_color = "#2ea043" if rec.get("passes_audit") else "#d29922"

                    st.markdown(f"""
                    <div style="background: rgba(22, 27, 34, 0.95); border: 1px solid {badge_color}; border-radius: 8px; padding: 12px; margin-bottom: 8px;">
                        <div style="display: flex; justify-content: space-between; align-items: center;">
                            <strong style="color: #79c0ff; font-size: 0.95rem;">Iteration {it_num}</strong>
                            <span style="background: {badge_color}; color: #ffffff; padding: 2px 8px; border-radius: 10px; font-size: 0.7rem; font-weight: bold;">{status_text}</span>
                        </div>
                        <div style="margin-top: 8px; font-size: 0.85rem; color: #c9d1d9; line-height: 1.5;">
                            🎯 <b>Grounding Score:</b> <span style="color: {'#7ee787' if g_score >= 85 else '#ffa657'}; font-weight: bold;">{g_score}%</span><br>
                            📑 <b>Citation Density:</b> {d_score}%<br>
                            ➕ <b>New Sources:</b> {rec.get('new_sources_retrieved', 0)}
                        </div>
                        <div style="margin-top: 8px; font-size: 0.78rem; color: #8b949e; background: rgba(13, 17, 23, 0.6); padding: 6px; border-radius: 4px;">
                            <b>Auditor Feedback:</b><br>{rec.get('feedback', 'No notes recorded.')[:150]}{'...' if len(rec.get('feedback', '')) > 150 else ''}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                    if rec.get("remedial_queries"):
                        st.caption(f"🔧 Remedial queries triggered: {', '.join(rec.get('remedial_queries', []))}")


    # Multi-Agent Communication Trace (Contract Net Protocol)
    if cnp_messages:
        with st.expander(f"📡 Multi-Agent Communication Trace ({len(cnp_messages)} ACL Messages)", expanded=False):
            st.caption("FIPA-ACL communication log showing Contract Net Protocol (CFP, PROPOSE, ACCEPT-PROPOSAL, INFORM, REQUEST)")
            for msg in cnp_messages:
                perf = msg.get("performative", "inform")
                sender = msg.get("sender", "Agent")
                receiver = msg.get("receiver", "Agent")
                timestamp = msg.get("timestamp", "")
                content = msg.get("content", "")

                # Badge mapping
                badge_class = "badge-inform"
                if perf == "cfp":
                    badge_class = "badge-cfp"
                elif perf == "propose":
                    badge_class = "badge-propose"
                elif perf == "accept-proposal":
                    badge_class = "badge-accept"
                elif perf == "request":
                    badge_class = "badge-request"

                # Content preview
                if isinstance(content, dict):
                    content_str = json.dumps(content, indent=2)
                else:
                    content_str = str(content)

                st.markdown(f"""
                <div style="background: rgba(13, 17, 23, 0.7); border-left: 3px solid #58a6ff; padding: 6px 12px; margin-bottom: 6px; border-radius: 4px;">
                    <span style="color: #8b949e; font-size: 0.75rem;">[{timestamp}]</span>&nbsp;
                    <span class="{badge_class}">{perf.upper()}</span>&nbsp;
                    <strong style="color: #c9d1d9; font-size: 0.85rem;">{sender}</strong>
                    <span style="color: #8b949e;">&nbsp;➔&nbsp;</span>
                    <strong style="color: #79c0ff; font-size: 0.85rem;">{receiver}</strong>
                    <div style="color: #8b949e; font-size: 0.8rem; margin-top: 4px;">
                        <code>{content_str[:180]}{'...' if len(content_str) > 180 else ''}</code>
                    </div>
                </div>
                """, unsafe_allow_html=True)

    # Collapsible Plan
    with st.expander("🔍 Awarded Contract Queries & Sub-questions"):
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Academic Literature Specialist (arXiv):**")
            for q in plan_data.get("arxiv_queries", []):
                st.code(q, language="text")
        with c2:
            st.markdown("**Industry Intelligence Scout (Web):**")
            for q in plan_data.get("web_queries", []):
                st.code(q, language="text")
        if plan_data.get("sub_questions"):
            st.markdown("**Decomposed Sub-questions:**")
            for sub in plan_data.get("sub_questions", []):
                st.markdown(f"- {sub}")

    # Collapsible Sources
    with st.expander(f"📖 Retrieved & Filtered Evidence Base ({len(sources)} sources)"):
        for idx, src in enumerate(sources, 1):
            stype = src.get("source_type", "")
            if "PDF" in stype or "Full-Text" in stype:
                badge_class = "badge-pdf"
            elif "ArXiv" in stype:
                badge_class = "badge-arxiv"
            else:
                badge_class = "badge-web"
            retriever = src.get("retrieved_by", "Worker")
            st.markdown(f"""
            **[{idx}] [{src.get('title', 'Untitled')}]({src.get('url', '#')})**  
            <span class="{badge_class}">{src.get('source_type', 'Web')}</span> | Agent: <code>{retriever}</code> | Date: `{src.get('published', 'N/A')}` | Score: `{src.get('score', 0):.2f}`  
            _{src.get('content', '')[:250]}..._
            """, unsafe_allow_html=True)
            st.divider()

    # Report Output Tabs
    st.markdown("### 📊 Synthesized Research Report")
    tab1, tab2, tab3 = st.tabs(["📄 Final Report (Rendered)", "📝 Raw Markdown", "📑 Citation Index"])
    
    with tab1:
        st.markdown(final_report)
        
    with tab2:
        st.code(final_report, language="markdown")
        
    with tab3:
        st.markdown("### Grounded Source References")
        for idx, src in enumerate(sources, 1):
            st.markdown(f"**[{idx}]** [{src.get('title', 'Untitled')}]({src.get('url', '#')}) — *{src.get('source_type', 'Web')} ({src.get('published', 'N/A')})*")

    # Download Button
    topic_slug = "".join([c if c.isalnum() else "_" for c in active.get("objective", "research")[:30]])
    st.download_button(
        label="📥 Download Research Dossier (.md)",
        data=final_report,
        file_name=f"{topic_slug}_report.md",
        mime="text/markdown",
        use_container_width=True
    )

    st.markdown("---")

    # ==========================================
    # SHORT-TERM WORKING MEMORY: FOLLOW-UP Q&A
    # ==========================================
    st.markdown("### 💬 Interactive Research Dialogue (Working Memory)")
    st.caption("Context-retaining conversational buffer — ask follow-up questions, drill down into citations, or compare concepts.")

    # Render Dialogue Turns
    if chat_history:
        for turn in chat_history:
            role = turn.get("role", "user")
            content = turn.get("content", "")
            with st.chat_message("user" if role == "user" else "assistant"):
                st.markdown(content)
    else:
        st.info("💡 **Working Memory is ready.** Ask follow-up questions like *'Explain section 2 in simpler terms'*, *'What hardware limits did source [1] mention?'*, or *'What are the open challenges?'*")

    # Follow-Up Chat Input
    followup_query = st.chat_input("Ask a follow-up question or request a drill-down into this research...")
    if followup_query:
        # Instantiate agent hydrated with existing session & short-term memory
        agent = AutonomousResearchAgent(
            model_name=model_choice,
            session_id=sid
        )
        # Hydrate agent's working memory with existing chat history
        agent.short_term_memory.load_from_list(chat_history)

        with st.spinner("🧠 Director reasoning over report & working memory..."):
            answer = agent.chat_followup(
                user_question=followup_query,
                report_context=final_report,
                sources=sources
            )

        # Update session state with refreshed dialogue history
        updated_history = agent.short_term_memory.get_history()
        st.session_state.active_research["chat_history"] = updated_history
        long_term_mem.update_chat_history(sid, updated_history)
        st.rerun()

    # Memory controls
    c_clear, _ = st.columns([1, 4])
    with c_clear:
        if chat_history and st.button("🧹 Clear Working Dialogue", use_container_width=True):
            st.session_state.active_research["chat_history"] = []
            long_term_mem.update_chat_history(sid, [])
            st.rerun()
