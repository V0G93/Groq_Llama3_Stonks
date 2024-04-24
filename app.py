import streamlit as st
import pandas as pd
import os
from dotenv import load_dotenv
from crewai import Agent, Task, Crew
from langchain_groq import ChatGroq
load_dotenv()

def main():

    # Set up the customization options
    st.sidebar.title('Customization')
    model = st.sidebar.selectbox(
        'Choose a model',
        ['mixtral-8x7b-32768', 'llama2-70b-4096','llama3-70b-8192']
    )

    llm = ChatGroq(
            temperature=0.01, 
            groq_api_key = os.getenv('GROQ_API_KEY'), 
            model_name=model
        )

    # Streamlit UI
    st.title('Stonks Assistant')
    st.image('stonks.webp')
    multiline_text = """
    Stonks Investment Assistant is designed to guide users through identifying and analyzing market trends in stocks and cryptocurrencies. It leverages a team of AI agents, each with a specific role, to clarify queries, evaluate market data, recommend investment strategies, and provide real-time analysis.
    """

    st.markdown(multiline_text, unsafe_allow_html=True)

    Market_Trend_Analysis_Agent = Agent(
    role='Market_Trend_Analysis_Agent',
    goal="""analyze current market trends based on historical and real-time data of stocks and cryptocurrencies.""",
    backstory="""You are an expert in market trend analysis. Your goal is to identify significant patterns and movements that could indicate investment opportunities.""",
    verbose=True,
    allow_delegation=False,
    llm=llm,
    )

    Data_Visualization_Agent = Agent(
        role='Data_Visualization_Agent',
        goal="""visualize data of stocks and cryptocurrencies to provide a clear and comprehensible representation of trends and volatilities.""",
        backstory="""Specialized in financial data visualization, your task is to transform complex data sets into clear, actionable charts for the user.""",
        verbose=True,
        allow_delegation=False,
        llm=llm,
    )

    Investment_Strategy_Recommendation_Agent = Agent(
        role='Investment_Strategy_Recommendation_Agent',
        goal="""suggest investment strategies based on trend analysis and the user's risk profile.""",
        backstory="""As an expert in investment strategies, you recommend approaches that best align with the user's goals and risk tolerance.""",
        verbose=True,
        allow_delegation=False,
        llm=llm,
    )

    Financial_Report_Generator_Agent = Agent(
        role='Financial_Report_Generator_Agent',
        goal="""generate detailed reports including trend analysis, visualizations, and recommended investment strategies.""",
        backstory="""You are a financial report wizard, capable of synthesizing data and analysis into comprehensive, investor-ready reports.""",
        verbose=True,
        allow_delegation=False,
        llm=llm,
    )

    user_query = st.text_input("Describe your investment interest or query:")

    if user_query:

        task_analyze_trends = Task(
            description=f"""Analyze market trends for the query: '{user_query}'""",
            agent=Market_Trend_Analysis_Agent,
            expected_output="A detailed analysis of relevant market trends."
        )

        task_visualize_data = Task(
            description="Visualize the data related to the provided query.",
            agent=Data_Visualization_Agent,
            expected_output="Data visualizations reflecting current and past trends."
        )

        task_recommend_strategy = Task(
            description="Recommend investment strategies based on the conducted analysis.",
            agent=Investment_Strategy_Recommendation_Agent,
            expected_output="Recommended investment strategies that match the user's risk profile."
        )

        task_generate_report = Task(
            description="Generate a financial report based on the findings from the analyses and visualizations.",
            agent=Financial_Report_Generator_Agent,
            expected_output="A comprehensive financial report including analysis, visualizations, and investment strategies."
        )
        crew = Crew(
            agents=[Market_Trend_Analysis_Agent, Data_Visualization_Agent, Investment_Strategy_Recommendation_Agent, Financial_Report_Generator_Agent],
            tasks=[task_analyze_trends, task_visualize_data, task_recommend_strategy, task_generate_report],
            verbose=2
        )

        result = crew.kickoff()

        st.write(result)

if __name__ == "__main__":
    main()