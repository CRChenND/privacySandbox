import os
import configparser
import openai
import warnings

from langchain.chains import LLMChain
from langchain.prompts.chat import ChatPromptTemplate, HumanMessagePromptTemplate
from langchain.prompts import PromptTemplate, FewShotPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from langchain.output_parsers import PydanticOutputParser

import json
from datetime import datetime
from random import randint

warnings.filterwarnings("ignore", category=DeprecationWarning)


# Set up configuration. Remember to input your OPENAI_API_KEY in config.ini file
config = configparser.ConfigParser()
config.read('config.ini')

os.environ['OPENAI_API_KEY'] = config['DEFAULT']['OPENAI_API_KEY']


# load prompts and few shot examples
with open('prompts.json') as prompts_file:
    prompts = json.load(prompts_file)

with open('examples.json') as examples_file:
    fewshot_examples = json.load(examples_file)

class EventSet(BaseModel):
    event: list[str] = Field(description='A list of calendar event content')

class ScheduleEntry(BaseModel):
    start_time: str = Field(description='The start time of a schedule event in the format of YYYY-MM-DD hh:mm:ss')
    end_time: str = Field(description='The end time of a schedule event in the format of YYYY-MM-DD hh:mm:ss')
    event: str = Field(description='A schedule event in the calendar')
    address: str = Field(description='The place where the event happen, must including four parts: street, city, state, and zipcode')
    latitude: float = Field(description='The corresponding latitude of the address')
    longitude: float = Field(description='The corresponding longitude of the address')

class Schedule(BaseModel):
    schedule: list[ScheduleEntry]


class Generator:
    def __init__(self):
        self.persona_profile = None
        self._event_sets = None
        self.schedule = None

    def get_persona_profile(self, guidance) -> str:
        '''
        Generate the persona description from LLM

        Parameters:
            guidance(str): The text briefly describes a persona.

        Return	
            str: persona description
        '''
        examples = fewshot_examples['profile']

        example_prompt = PromptTemplate(
            input_variables=['persona'],
            template=prompts["profile"]["prompt"]
        )

        few_shot_prompt = FewShotPromptTemplate(
            examples=examples,
            example_prompt=example_prompt,
            prefix=prompts["profile"]["prefix"],
            suffix=prompts["profile"]["suffix"],
            input_variables=['guidance', "template"],
            example_separator="\n\n"
        )

        chain = LLMChain(llm=ChatOpenAI(model_name='gpt-3.5-turbo',temperature=0.9),
                         prompt=few_shot_prompt)
        persona_profile = chain.invoke(input={'guidance':guidance, "template":prompts["profile"]["template"]})
        self.persona_profile = persona_profile["text"]

        return self.persona_profile
    

    def _gen_events(self, profile: str) -> list:
        '''
        Generate the persona's event sets to support generate the schedule.

        Input
            profile(str): the profile description of the persona to support generate its weekly event sets.

        Return
            list: a event set for the input persona.
        '''
        event_parser = PydanticOutputParser(pydantic_object=EventSet)

        human_prompt = HumanMessagePromptTemplate.from_template("{request}\n{profile}\n{format_instructions}")
        chat_prompt = ChatPromptTemplate.from_messages([human_prompt])
        
        request = chat_prompt.format_prompt(
            profile = profile,
            request = prompts["event"]["prompt"],
            format_instructions = event_parser.get_format_instructions()
        ).to_messages()

        model = ChatOpenAI(model_name='gpt-3.5-turbo',temperature=0.9)
        results = model(request)
        results_value = event_parser.parse(results.content)
        self._event_sets = results_value

        return self._event_sets
    
    def get_schedule(self, persona: str, start_date: str, end_date: str) -> list:
        '''
        Generate the persona's schedule.

        Input
            person(str): the profile description of the persona,
            start_date(str): the start date of the schedule in the format of YYYY-MM-DD
            end_date(str): the end date of the schedule in the format of YYYY-MM-DD

        Return
            list: the persona's schedule
        '''
       
        # We need to add .replace("{", "{{").replace("}", "}}") after serialising as JSON so that the curly brackets in the JSON won’t be mistaken for prompt variables by LangChain
        curated_examples = []
        for i,v in enumerate(fewshot_examples['schedule']):
            curated_examples.append({"persona": v["persona"], 
                                     "event_example":v["event_example"], 
                                     "schedule_example":[]})
            for entry in v["schedule_example"]:
                curated_examples[i]["schedule_example"].append(ScheduleEntry.model_validate(entry).model_dump_json().replace("{", "{{").replace("}", "}}"))

        examples = curated_examples

        schedule_parser = PydanticOutputParser(pydantic_object=Schedule)
        
        example_prompt = PromptTemplate(
            input_variables=['persona', 'event_example', 'schedule_example'],
            template=prompts["schedule"]["prompt"],
        )

        few_shot_prompt = FewShotPromptTemplate(
            examples=examples,
            example_prompt=example_prompt,
            prefix=prompts["schedule"]["prefix"],
            suffix=prompts["schedule"]["suffix"],
            input_variables=['persona', "start_date", "end_date", "event_example"],
            partial_variables={"format_instructions": schedule_parser.get_format_instructions()},
        )

        chain = LLMChain(llm=ChatOpenAI(model_name='gpt-3.5-turbo-16k',temperature=0.5),
                         prompt=few_shot_prompt)
        generated_schedule = chain.invoke(input={
            'persona':persona, 
            'event_example': self._gen_events(persona),
            'start_date':start_date, 
            'end_date':end_date
        })

        self.persona_profile = generated_schedule['text']
        return self.persona_profile


if __name__=="__main__":
    gen = Generator()
    # test case 1: generate persona
    persona = gen.get_persona_profile(guidance = "A 23-year-old Asian female.")
    print(persona)

    # test case 2: generate weekly schedule
    schedule = gen.get_schedule(persona, "2024-01-05", "2024-01-11")
    print(schedule)