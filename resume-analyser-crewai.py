
# import the required libraries

import os
from google.colab import userdata  
import fitz
import gradio as gr
from crewai import Agent, Task, Crew, LLM, Process
import re
from dotenv import load_dotenv
    

# Load (and expose) your Gemini API key in the environment

try:
    os.environ["GEMINI_API_KEY"] = os.getenv("GOOGLE_API_KEY")
except Exception as e:
    raise RuntimeError(
        "Colab error: Could not load the Gemini key. Make sure "
        "'GOOGLE_API_KEY' exists in Secrets and is non‑empty."
    )


#  Build a litellm-compatible CrewAI LLM instance

gemini_llm = LLM(
    model="gemini/gemini-1.5-flash",
    api_key=os.getenv("GEMINI_API_KEY"),
    temperature=0.1,
    verbose=False  # Set to False to reduce logs
)


#  PDF parsing utility

def extract_text_from_pdf(file_path: str) -> str:
    txt = ""
    try:
        doc = fitz.open(file_path)
        for page in doc:
            txt += page.get_text()
        doc.close()
    except Exception as pdf_err:
        raise RuntimeError(f"Failed to read PDF {file_path}: {pdf_err}") from pdf_err
    return txt



# Agent and Task builders

def get_reader_agent() -> Agent:
    return Agent(
        role="Resume Reader",
        goal="Extract key skills, education, and work experience from a resume",
        backstory="You are an expert resume analyst, accurate and concise.",
        verbose=False,  # Reduce verbosity
        llm=gemini_llm,
        allow_delegation=False
    )


def get_read_task(resume_path: str, agent: Agent) -> Task:
    resume_text = extract_text_from_pdf(resume_path)
    return Task(
        description=(
            "Extract the candidate's key skills, education, and professional experience "
            "from the following résumé text:\n\n" + resume_text
        ),
        expected_output="A structured list of skills, education history, and job experiences.",
        agent=agent,
    )


def get_evaluator_agent() -> Agent:
    return Agent(
        role="Job Match Evaluator",
        goal="Evaluate how well a candidate's résumé aligns with a job posting",
        backstory="You are a hiring analyst with domain knowledge in HR evaluation.",
        verbose=False,  # Reduce verbosity
        llm=gemini_llm,
        allow_delegation=False
    )


def get_eval_task(job_desc: str, agent: Agent) -> Task:
    return Task(
        description=(
            "Given the extracted résumé data from the previous step, evaluate how well "
            "the candidate fits the following job description:\n\n" + job_desc +
            "\n\nProvide a comprehensive analysis with the following format:\n\n"
            "**Match Score: XX/100**\n\n"
            "**Detailed Analysis:**\n\n"
            "**✅ Strengths & Matching Qualifications:**\n"
            "- **Relevant Skills:** [List matching technical and soft skills]\n"
            "- **Strong Projects:** [Highlight relevant projects and their impact]\n"
            "- **Relevant Experience:** [Match with job requirements and responsibilities]\n"
            "- **Academic Background:** [Education alignment with role requirements]\n\n"
            "**⚠️ Areas for Improvement:**\n"
            "- **Missing Skills:** [Technical skills or certifications needed]\n"
            "- **Experience Gaps:** [Areas where more experience would help]\n"
            "- **Recommendations:** [Specific steps to strengthen profile]\n\n"
            "**🎯 Alternative Job Position Suggestions:**\n"
            "Based on the candidate's profile, suggest 3-4 alternative positions that might be a better fit:\n"
            "- **Position 1:** [Job title] - [Why it's a good fit] - [Match percentage]\n"
            "- **Position 2:** [Job title] - [Why it's a good fit] - [Match percentage]\n"
            "- **Position 3:** [Job title] - [Why it's a good fit] - [Match percentage]\n\n"
            "**🚀 Career Growth Scope & Trajectory:**\n"
            "- **Short-term (1-2 years):** [Immediate growth opportunities and skill development]\n"
            "- **Medium-term (3-5 years):** [Career progression paths and potential roles]\n"
            "- **Long-term (5+ years):** [Leadership opportunities and specialization areas]\n"
            "- **Industry Trends:** [Relevant market trends affecting career growth]\n\n"
            "**💡 Strategic Recommendations:**\n"
            "- **Immediate Actions:** [What to do in next 3-6 months]\n"
            "- **Skill Development:** [Priority skills to learn or improve]\n"
            "- **Network Building:** [Industry connections to make]\n"
            "- **Portfolio Enhancement:** [Projects or certifications to pursue]"
        ),
        expected_output="A comprehensive career analysis with match score, alternative positions, growth scope, and strategic recommendations in markdown format.",
        agent=agent,
    )



#  Extract clean output from CrewAI result

def extract_final_output(crew_result):
    """Extract the final agent output from CrewAI result object"""
    try:
        # Try to get the raw output from the result
        if hasattr(crew_result, 'raw'):
            return crew_result.raw
        elif hasattr(crew_result, 'result'):
            return crew_result.result
        elif hasattr(crew_result, 'output'):
            return crew_result.output
        else:
            # Convert to string and extract meaningful content
            result_str = str(crew_result)
            return result_str
    except Exception as e:
        return f"Error extracting result: {str(e)}"


def format_output_for_gradio(raw_output):
    """Format the output for better display in Gradio"""
    try:
        # Clean up any remaining log traces
        output = str(raw_output)

        # Remove common log patterns
        log_patterns = [
            r'\[.*?\]\s*',  # Remove timestamp patterns
            r'INFO:.*?\n',  # Remove INFO logs
            r'DEBUG:.*?\n', # Remove DEBUG logs
            r'Agent:.*?Executing Task.*?\n',  # Remove execution logs
            r'Status:.*?\n',  # Remove status updates
        ]

        for pattern in log_patterns:
            output = re.sub(pattern, '', output, flags=re.MULTILINE)

        # Clean up extra whitespace
        output = re.sub(r'\n\s*\n', '\n\n', output)
        output = output.strip()

        # If output doesn't look like markdown, try to structure it
        if not ('**' in output or '#' in output):
            # Try to identify score pattern
            score_match = re.search(r'(\d+)/100', output)
            if score_match:
                score = score_match.group(1)
                # Structure the output
                formatted_output = f"""## 📊 Resume Analysis Results

### **Match Score: {score}/100**

### **Detailed Evaluation:**
{output}

---
*Analysis completed using AI-powered resume evaluation*"""
                return formatted_output

        return output

    except Exception as e:
        return f"Error formatting output: {str(e)}\n\nRaw output:\n{raw_output}"


# Executing the Crew


def run_resume_analysis(resume_path: str, job_description: str):
    reader_agent = get_reader_agent()
    eval_agent = get_evaluator_agent()

    read_task = get_read_task(resume_path, reader_agent)
    eval_task = get_eval_task(job_description, eval_agent)

    crew = Crew(
        agents=[reader_agent, eval_agent],
        tasks=[read_task, eval_task],
        process=Process.sequential,
        verbose=False  # Reduce crew verbosity
    )

    result = crew.kickoff()

    # Extract and format the clean output
    clean_output = extract_final_output(result)
    formatted_output = format_output_for_gradio(clean_output)

    return formatted_output


#  Gradio interface


def analyze_resume_interface(resume_file, job_desc):
    if resume_file is None or not job_desc.strip():
        return "⚠️ Please upload a résumé (PDF) and paste a job description."

    path = resume_file.name
    try:
        # Show loading message
        loading_msg = """## 🔄 Analysis in Progress...

Please wait while our AI agents analyze your resume:

1. **Resume Reader**: Extracting skills, education, and experience...
2. **Job Match Evaluator**: Comparing with job requirements...

This may take 30-60 seconds depending on resume complexity."""

        yield loading_msg

        # Run the analysis
        result = run_resume_analysis(path, job_desc)

        # Return the formatted result
        yield result

    except Exception as e:
        error_msg = f"""## ❌ Analysis Error

**Error Type:** {type(e).__name__}

**Details:** {str(e)}

Please try again or check your resume file format."""
        yield error_msg


#  Gradio UI


with gr.Blocks(
    theme=gr.themes.Soft(),
    title="Smart Resume Analyzer",
    css="""
    .gradio-container {
        max-width: 1200px !important;
    }
    """
) as demo:

    gr.Markdown("""
    # 🧠 Smart Resume Analyzer using CrewAI

    Upload your resume and job description to get an AI-powered compatibility analysis with detailed feedback.
    """)

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 📄 Resume Upload")
            resume_input = gr.File(
                label="Upload Résumé (PDF)",
                file_types=[".pdf"],
                height=100
            )

            gr.Markdown("### 💼 Job Description")
            job_input = gr.Textbox(
                lines=8,
                label="Paste the job description here",
                placeholder="Paste the complete job description including requirements, responsibilities, and qualifications..."
            )

            submit_btn = gr.Button(
                "🚀 Analyze Resume",
                variant="primary",
                size="lg"
            )

        with gr.Column(scale=2):
            gr.Markdown("### 📊 Analysis Results")
            output_box = gr.Markdown(
                value="Upload a resume and job description, then click 'Analyze Resume' to get started.",
                height=600
            )

    # Connect the interface
    submit_btn.click(
        fn=analyze_resume_interface,
        inputs=[resume_input, job_input],
        outputs=output_box,
        show_progress=True
    )

    gr.Markdown("""
    ---
    **How it works:**
    1. **Resume Reader Agent** extracts skills, education, and experience from your PDF
    2. **Job Match Evaluator Agent** compares your profile against job requirements
    3. You get a detailed compatibility score with actionable feedback
    """)

# Launch the demo
demo.launch(debug=True, share=True)