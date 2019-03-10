# chatbot_demo
from rasa_core import utils
from rasa_core.agent import Agent
from rasa_core.policies.keras_policy import KerasPolicy
from rasa_core.policies.memoization import MemoizationPolicy
# NLU - NATURAL LANGUAGE UNDERSTANDING IN RASA/// IDENTIFYING THE INTENT BEHIND LANGUAGE
intent_data = """
## intent:introduction
- how are you?
- howdy?
- h r u?
- hi
- hey
- hellow

## intent:ask_time
- whats the time now?
- can u tell me the time now
- time now plese?
- tell me the time now

## intent:ask_weather
- whats the temperature in Bngalore now
- tell the temperature of Bangalore
- how is the weather condition in Bangalore now

## intent:ask_hashtags
- what is the tending hashtags today
- today trending hashtags please
- tell me what is trending today
- list top 10 trending hashtags
"""
%store intent_data > data.md  ## writing in a data file
import spacy
spacy.load('en')

setting="""
pipeline: spacy_sklearn
"""

%store setting > setting.yml
!pip install sklearn_crfsuite
from rasa_nlu.training_data import load_data
from rasa_nlu import config
from rasa_nlu.model import Trainer
from rasa_nlu.model import Interpreter
trainer=Trainer(config.load('setting.yml'))
training_data=load_data('data.md')
trainer.train(training_data)

--------
model_directory=trainer.persist('models/',fixed_model_name="current")
interpreter=Interpreter.load('models/default/current')
--------
domain_text = """

intents:
- introduction
- ask_time
- ask_weather
- ask_hashtags

actions:
- utter_introduce
- utter_time
- utter_weather
- utter_hashtags

templates:
 utter_introduce:
   - Welcome
   - hi. nice to meet u
   - hellow, hope you are doing good
 utter_time:
   - your time is really bad
   - your time is really good
 utter_weather:
   - its cloudy
   - its sunny
   - why do u care
 utter_hashtags:
   - Indvsaus,welcomeModi,abhinavisGreat
   - womensday
"""
%store domain_text > domain.yml
-------
stories = """
## introduce
* introduction
- utter_introduce

## Ask Time
* ask_time
- utter_time

## Ask Weather
* ask_weather
- utter_weather

## Ask Hashtags
* ask_hashtags
- utter_hashtags
"""

%store stories > stories.md
--------
domain_file="domain.yml"
model_path="models/dialogue"
training_data_file="stories.md"    
agent = Agent(
    domain_file,
    policies=[MemoizationPolicy(max_history=3), KerasPolicy()]
    )
training_data = agent.load_data(training_data_file)
agent.train(
    training_data,
    )
agent.persist(model_path)

----------------
#creat a new .py file put the bellow code in it and run the file one the command prompt to interact with the chatbot
from rasa_core import utils
from rasa_core.agent import Agent
from rasa_core.interpreter import RasaNLUInterpreter, NaturalLanguageInterpreter
import rasa_core
from rasa_core import run

interpreter = RasaNLUInterpreter("models/default/current")
agent = Agent.load("models/dialogue", interpreter=interpreter)

def run_bot(dbug=False):
    if dbug:
        init_debug_logging()
    interpreter = NaturalLanguageInterpreter.create("models/default/current")
    from rasa_core.utils import EndpointConfig
    action_endpoint = EndpointConfig(url="http://localhost:5056/webhook")
    agent = Agent.load("models/dialogue", interpreter=interpreter,action_endpoint=action_endpoint)
    rasa_core.run.serve_application(agent,channel='cmdline')
run_bot()
