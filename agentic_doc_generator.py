from typing import TypedDict, Optional, List
import os
import logging
import ollama
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('doc_generator.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Configuration
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3")
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.1"))
MAX_REVISION_ATTEMPTS = 3


class DocumentState(TypedDict):
    topic: str
    requirements: str
    plan: Optional[str]
    content: Optional[str]
    review_feedback: Optional[str]
    revision_count: int
    is_approved: bool
    messages: List[dict]


def load_system_prompt() -> str:
    try:
        base_dir = os.path.dirname(__file__)
        with open(os.path.join(base_dir, 'system_prompt.txt'), 'r') as f:
            return f.read()
    except FileNotFoundError:
        logger.error("system_prompt.txt not found")
        return "You are an expert document writer."


def load_documentation_details() -> str:
    try:
        base_dir = os.path.dirname(__file__)
        with open(os.path.join(base_dir, 'documentation_details.txt'), 'r') as f:
            return f.read()
    except FileNotFoundError:
        logger.error("documentation_details.txt not found")
        return "Create comprehensive documentation."


def call_llm(messages: List[dict]) -> str:
    try:
        logger.info(f"Calling LLM with {len(messages)} messages")
        response = ollama.chat(
            model=OLLAMA_MODEL,
            messages=messages
        )
        return response['message']['content']
    except Exception as e:
        logger.error(f"Error calling LLM: {e}")
        return "I'm sorry, I'm having trouble processing your request right now."


def planning_agent(state: DocumentState) -> DocumentState:
    logger.info("=== PLANNING AGENT STARTED ===")
    system_prompt = load_system_prompt()
    doc_details = load_documentation_details()
    planning_prompt = f"""
{system_prompt}

You are now in the PLANNING phase. Your task is to create a comprehensive plan for the documentation.

Documentation Details:
{doc_details}

Topic: {state['topic']}
Requirements: {state['requirements']}

Create a detailed plan that includes:
1. Document structure and sections
2. Content outline for each section
3. Key points to cover
4. Sequence considerations
5. Estimated content length for each section
6. Any special requirements or considerations

Format your response as a structured plan with clear sections and subsections.
"""
    messages = [
        {'role': 'system', 'content': planning_prompt}
    ]
    plan = call_llm(messages)
    logger.info("Planning agent completed successfully")
    return {
        **state,
        'plan': plan,
        'messages': state["messages"] + [{"role": "assistant", "content": f"PLAN CREATED:\n{plan}"}]
    }


def writing_agent(state: DocumentState) -> DocumentState:
    logger.info("=== WRITING AGENT STARTED ===")
    system_prompt = load_system_prompt()
    doc_details = load_documentation_details()
    writing_prompt = f"""
{system_prompt}

You are now in the WRITING phase. Your task is to create the actual document content based on the plan.

Documentation Details:
{doc_details}

Topic: {state['topic']}
Requirements: {state['requirements']}
Plan: {state['plan']}

Write comprehensive documentation that:
1. Follows the plan structure exactly
2. Includes all required sections from documentation_details.txt
3. Uses clear, professional language
4. Provides practical examples and step-by-step instructions
5. Uses proper formatting with headers, lists, and emphasis
6. Is comprehensive yet user-friendly

Write the complete document content now.
"""
    messages = [
        {'role': 'system', 'content': writing_prompt}
    ]
    content = call_llm(messages)
    logger.info("Writing agent completed successfully")
    return {
        **state,
        'content': content,
        'messages': state["messages"] + [{"role": "assistant", "content": f"CONTENT WRITTEN:\n{content}"}]
    }


def review_agent(state: DocumentState) -> DocumentState:
    logger.info("=== REVIEW AGENT STARTED ===")
    system_prompt = load_system_prompt()
    doc_details = load_documentation_details()
    review_prompt = f"""
{system_prompt}

You are now in the REVIEW phase. Your task is to thoroughly review the document and provide feedback.

Documentation Details:
{doc_details}

Topic: {state['topic']}
Requirements: {state['requirements']}
Plan: {state['plan']}
Content: {state['content']}

Review the document for:
1. Completeness - Does it cover all required sections?
2. Accuracy - Is the information correct and up-to-date?
3. Clarity - Is the language clear and understandable?
4. Structure - Is the organization logical and user-friendly?
5. Professionalism - Is the tone appropriate and consistent?
6. Compliance - Does it meet all requirements from documentation_details.txt?

Provide detailed feedback with specific suggestions for improvement.
If the document is perfect, respond with "APPROVED - NO REVISIONS NEEDED."
Otherwise, provide specific revision recommendations.
"""
    messages = [
        {'role': 'system', 'content': review_prompt}
    ]
    feedback = call_llm(messages)
    is_approved = "APPROVED - NO REVISIONS NEEDED" in feedback.upper()
    logger.info(f"Review agent completed. Approved: {is_approved}")
    return {
        **state,
        'review_feedback': feedback,
        'is_approved': is_approved,
        'messages': state["messages"] + [{"role": "assistant", "content": f"REVIEW FEEDBACK:\n{feedback}"}]
    }


def revision_agent(state: DocumentState) -> DocumentState:
    logger.info("=== REVISION AGENT STARTED ===")
    system_prompt = load_system_prompt()
    doc_details = load_documentation_details()
    revision_prompt = f"""
{system_prompt}

You are now in the REVISION phase. Your task is to revise the document based on the review feedback.

Documentation Details:
{doc_details}

Topic: {state['topic']}
Requirements: {state['requirements']}
Plan: {state['plan']}
Original Content: {state['content']}
Review Feedback: {state['review_feedback']}

Revise the document to address all issues mentioned in the review feedback.
Make sure to:
1. Address all specific points mentioned in the feedback
2. Maintain the original structure and plan
3. Improve clarity, completeness, and accuracy
4. Ensure all requirements are met
5. Keep the professional tone and style

Provide the revised document content.
"""
    messages = [
        {'role': 'system', 'content': revision_prompt}
    ]
    revised_content = call_llm(messages)
    logger.info("Revision agent completed successfully")
    return {
        **state,
        'content': revised_content,
        'revision_count': state["revision_count"] + 1,
        'messages': state["messages"] + [{"role": "assistant", "content": f"REVISED CONTENT:\n{revised_content}"}]
    }


def workflow_router(state: DocumentState) -> str:
    logger.info("=== WORKFLOW ROUTER ===")
    if not state.get('plan'):
        logger.info("Routing to planning agent")
        return "planning"
    if not state.get('content'):
        logger.info("Routing to writing agent")
        return "writing"
    if not state.get('review_feedback'):
        logger.info("Routing to review agent")
        return "review"
    if state.get('is_approved'):
        logger.info("Document approved, ending workflow")
        return "end"
    if state.get('revision_count', 0) >= MAX_REVISION_ATTEMPTS:
        logger.info(f"Maximum revision attempts ({MAX_REVISION_ATTEMPTS}) reached, ending workflow")
        return "end"
    logger.info("Routing to revision agent")
    return "revision"


def generate_pdf(content: str, filename: str = "generated_documentation.pdf"):
    logger.info("=== PDF GENERATION STARTED ===")
    try:
        doc = SimpleDocTemplate(filename, pagesize=letter)
        styles = getSampleStyleSheet()
        story = []
        header_style = ParagraphStyle(
            'CustomHeader',
            parent=styles['Heading1'],
            fontSize=16,
            spaceAfter=12,
            spaceBefore=12
        )
        sections = content.split('\n\n')
        for section in sections:
            if section.strip():
                if section.strip().startswith('#') or section.strip().isupper():
                    header_text = section.strip().replace('#', '').strip()
                    story.append(Paragraph(header_text, header_style))
                else:
                    story.append(Paragraph(section, styles['Normal']))
                    story.append(Spacer(1, 6))
        doc.build(story)
        logger.info(f"PDF generated successfully: {filename}")
        return filename
    except Exception as e:
        logger.error(f"Error generating PDF: {e}")
        return None


def end_workflow(state: DocumentState) -> DocumentState:
    logger.info("=== ENDING WORKFLOW ===")
    if state.get('content'):
        pdf_filename = generate_pdf(state['content'])
        if pdf_filename:
            logger.info(f"Document generation completed successfully. PDF saved as: {pdf_filename}")
        else:
            logger.error("PDF generation failed")
    else:
        logger.error("No content to generate PDF from")
    return state


def create_workflow() -> StateGraph:
    workflow = StateGraph(DocumentState)
    workflow.add_node("planning", planning_agent)
    workflow.add_node("writing", writing_agent)
    workflow.add_node("review", review_agent)
    workflow.add_node("revision", revision_agent)
    workflow.add_node("end", end_workflow)
    workflow.set_entry_point("planning")
    workflow.add_conditional_edges(
        "planning",
        workflow_router,
        ["writing", "review", "revision", "end"]
    )
    workflow.add_conditional_edges(
        "writing",
        workflow_router,
        ["review", "revision", "end"]
    )
    workflow.add_conditional_edges(
        "review",
        workflow_router,
        ["revision", "end"]
    )
    workflow.add_conditional_edges(
        "revision",
        workflow_router,
        ["review", "end"]
    )
    return workflow.compile()


def main():
    logger.info("=== DOCUMENT GENERATION WORKFLOW STARTED ===")
    initial_state: DocumentState = {
        'topic': "User Manual for Customer Management System",
        'requirements': "Create a comprehensive user manual for a customer management system with features for adding, editing, and managing customer records",
        'plan': None,
        'content': None,
        'review_feedback': None,
        'revision_count': 0,
        'is_approved': False,
        'messages': []
    }
    workflow = create_workflow()
    checkpointer = MemorySaver()
    final_state = workflow.invoke(
        initial_state,
        config={"thread_id": "doc_generation"}
    )
    logger.info("=== DOCUMENT GENERATION WORKFLOW COMPLETED ===")
    return final_state


if __name__ == "__main__":
    main()
