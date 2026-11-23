"""
Autonomous Quantum Computing Research Agent (MVP CLI Prototype)
Tech Stack: DuckDuckGo Search, ArXiv API, ChromaDB/SentenceTransformers, Gemini API
"""

import os
import sys
from dotenv import load_dotenv

# Ensure local imports work cleanly
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.agent import QuantumResearchAgent, _extract_text

# Reconfigure stdout for UTF-8 compatibility on Windows terminals
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

def main():
    load_dotenv()
    
    objective = "Research the latest developments in Quantum Computing and prepare a summarized report."
    if len(sys.argv) > 1:
        objective = " ".join(sys.argv[1:])

    print("========================================================================")
    print("           AUTONOMOUS AI RESEARCH AGENT (MVP)           ")
    print("========================================================================\n")
    print(f"[*] Research Objective: '{objective}'\n")

    try:
        agent = QuantumResearchAgent()
        
        def status_update(msg, step):
            print(f"[{step}/4] {msg}")

        plan, sources, draft, final_report = agent.run_pipeline(objective, progress_callback=status_update)
        final_report_str = _extract_text(final_report)

        print("\n" + "="*40 + " FINAL RESEARCH REPORT " + "="*40 + "\n")
        print(final_report_str)
        print("\n" + "="*100)

        output_filename = "Research_Report.md"
        with open(output_filename, "w", encoding="utf-8") as f:
            f.write(final_report_str)
            
        print(f"\n[✓] Research completed successfully! Saved to '{output_filename}'")
        
    except ValueError as ve:
        print(f"\n[!] Configuration Error: {ve}")
    except Exception as e:
        print(f"\n[!] Unexpected Error during research pipeline: {e}")

if __name__ == "__main__":
    main()
