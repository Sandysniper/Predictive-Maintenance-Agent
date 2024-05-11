import streamlit as st
from crewai import Agent, Task, Crew
import os
from langchain_groq import ChatGroq
from langchain.agents import load_tools
import pandas as pd

os.environ["SERPER_API_KEY"] = "018591824635f940322a07169b38956fa75da3e7"

model_agent = ChatGroq(
    api_key=os.getenv("GROQ_API_KEY"),
    model="llama3-8b-8192"
)

# Title and Introduction
st.title("Predictive Maintenance Agent")
st.write("Optimizing Equipment Uptime and Reducing Maintenance Costs!")

# Streamlit User Inputs
st.sidebar.header("Equipment Details:")

equipment_type = st.sidebar.selectbox(
    "Equipment Type",
    options=["Pump", "Motor", "Compressor", "Valve", "Other"]
)

equipment_age = st.sidebar.number_input("Equipment Age", min_value=0)

equipment_usage = st.sidebar.number_input("Equipment Usage (hours)", min_value=0)

sensor_data_file = st.sidebar.file_uploader("Sensor Data (CSV file)", type="csv")

# Additional user inputs
maintenance_history = st.sidebar.text_area("Maintenance History (optional)")

operating_environment = st.sidebar.text_area("Operating Environment (optional)")

usage_patterns = st.sidebar.text_area("Usage Patterns (optional)")

manufacturer_recommendations = st.sidebar.text_area("Manufacturer's Recommendations (optional)")

# Create User Input Dictionary
user_inputs = {
    "equipment_type": equipment_type,
    "equipment_age": equipment_age,
    "equipment_usage": equipment_usage,
    "sensor_data": None,
    "maintenance_history": maintenance_history,
    "operating_environment": operating_environment,
    "usage_patterns": usage_patterns,
    "manufacturer_recommendations": manufacturer_recommendations
}
# If sensor data file is uploaded, read it into a DataFrame and add it to the user inputs
if sensor_data_file is not None:
    sensor_data = pd.read_csv(sensor_data_file)
    user_inputs["sensor_data"] = sensor_data.to_dict()

# Agent Backstory and Role

predictive_maintenance_backstory = """
You are a predictive maintenance expert with extensive knowledge of mechanical equipment, sensor data analysis, and machine learning algorithms. 
You can analyze equipment data to identify early signs of wear, predict future failures, and recommend proactive maintenance actions to minimize downtime and maximize equipment lifespan.
"""

predictive_maintenance_agent = Agent(
    role="Predictive Maintenance Expert",
    goal="Analyze the provided equipment data to identify potential failure modes and recommend proactive maintenance actions. Consider the equipment's age, usage, and sensor data to make your recommendations.",
    backstory=predictive_maintenance_backstory,
    verbose=True,
    allow_delegation=False,
  
    llm=model_agent
)

# Tasks
analysis_task = Task(
    description=f"Analyze equipment data given by user : {user_inputs} .Analyse the user data and provide insights into potential failures and maintenance recommendations based on this.",
    expected_output="""Detailed analysis of equipment data  given by user, including failure predictions and maintenance recommendations with followwing informations 
        1. Potential Failure Modes: List the potential failure modes that the equipment may be at risk of, based on the analysis of the provided user data.
        2. Maintenance Recommendations: For each potential failure mode, recommend specific maintenance actions that should be taken to prevent or mitigate the failure.
        """,
    agent=predictive_maintenance_agent,
    async_execution=True
)

report_task = Task(
    description="Generate a comprehensive report summarizing the equipment analysis and maintenance recommendations.",
    expected_output="Report summarizing equipment condition, potential failures, and recommended maintenance actions",
    agent=predictive_maintenance_agent,
    context=[analysis_task],
    
)

# Crew
crew = Crew(
    agents=[predictive_maintenance_agent],
    tasks=[analysis_task, report_task],
    verbose=1
)

# Recommendation Button
if st.sidebar.button("Get Maintenance Recommendations"):
    st.header("Maintenance Recommendations")
    with st.spinner("Analyzing equipment data and generating recommendations..."):
      result = crew.kickoff()
   
    st.write(f"""
              Task : {report_task.output.raw_output}""")