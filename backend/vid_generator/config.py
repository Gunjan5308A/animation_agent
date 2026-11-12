from dotenv import load_dotenv
import os

load_dotenv()

api_key = os.getenv('API_KEY')
base_url = os.getenv('BASE_URL')
planner_model = os.getenv('Planner_model')
prompt_model = os.getenv('prompt')
sub_agents = os.getenv('sub_agents')
review_model = os.getenv('Review')
