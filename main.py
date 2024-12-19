from crewai import Agent, Task, Crew
from dotenv import load_dotenv
import os 

load_dotenv()

support_agent = Agent(
    role="Senior Support Representative",
    goal="Be the most friendly and helpful support representative in your team",
    backstory="You work  at deepAI and are now working on providing support to {customer}, a super important learner for your company. You need to make sure that {customer} is happy and satisfied with the support you provide",
    allow_delegation=False,
    verbose = True,
)

support_quality_assurance_agent = Agent(
    role="Support Quality Assurance",
    goal = "Get recognition for providing the best support quality assurance in your team",
    backstory=("You work at deepAI and are working on a request from {customer} ensuring that the support representative is providing the best support possible. You need to make sure that the support representative is providing the best support possible to {customer}"),
    verbose = True,
)

from crewai_tools import SerperDevTool, ScrapeWebsiteTool, WebsiteSearchTool

docs_scrape_tool = ScrapeWebsiteTool(website_url ="https://www.aimletc.com/multi-ai-agent-system-for-customer-support-automation-using-crewai/")
print(docs_scrape_tool)

inquiry_resolution = Task(
    description = (
        "{customer} just reached out with a super important ask:\n"
        "{inquiry}\n\n"
        "{person} from {customer} is the one that reached out. "
        "Make sure to use everything you know to provide the best support possible."
        "You must strive to provide a complete and accurate response to the customer's inquiry."
    ),
    expected_output = (
        "A detailed, informative response to the customer\s inquiry that addresses all aspects of their question.\n"
        "The response should include references to everything you used to find the answer, including external data."),
        tools = [docs_scrape_tool],
        agent = support_agent
    )

quality_assurance_review = Task(
    description=(
        "Review the response drafted by the Senior Support Representative for {customer} inquiry. Ensure the answer is accurate and adheres to high quality standards. Verify that all parts of the custsomer's inquiry have been addressed thoroughly. Check for references and sources used to find the information, ensuring the response is well supported. "
    ),
    expected_output = ("A final, detailed and informative response ready to be sent to the customer."),
    agent = support_quality_assurance_agent
)

crew = Crew(
    agents = [support_agent, support_quality_assurance_agent],
    tasks = [inquiry_resolution, quality_assurance_review],
    memory = False,
)

inputs = {
    "customer": "John Smith",
    "person": "John Smith",
    "inquiry": "How can I implement CrewAI in my customer support system?"
}

result = crew.kickoff(inputs=inputs)
