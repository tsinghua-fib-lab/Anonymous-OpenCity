from pycityagent.brain.memory import WorkingMemory

GPT_API_USED = False

def generate_prompt(curr_input, prompt_lib_file): 
  """
  Takes in the current input (e.g. comment that you want to classifiy) and 
  the path to a prompt file. The prompt file contains the raw str prompt that
  will be used, which contains the following substr: !<INPUT>! -- this 
  function replaces this substr with the actual curr_input to produce the 
  final promopt that will be sent to the GPT3 server. 
  ARGS:
    curr_input: the input we want to feed in (IF THERE ARE MORE THAN ONE
                INPUT, THIS CAN BE A LIST.)
    prompt_lib_file: the path to the promopt file. 
  RETURNS: 
    a str prompt that will be sent to OpenAI's GPT server.  
  """
  if type(curr_input) == type("string"): 
    curr_input = [curr_input]
  curr_input = [str(i) for i in curr_input]

  f = open(prompt_lib_file, "r")
  prompt = f.read()
  f.close()
  for count, i in enumerate(curr_input):   
    prompt = prompt.replace(f"!<INPUT {count}>!", i)
  if "<commentblockmarker>###</commentblockmarker>" in prompt: 
    prompt = prompt.split("<commentblockmarker>###</commentblockmarker>")[2]
  return prompt.strip()

# [ ] Batch prompt
async def generate_wake_up_hour_batch(agents, prompt):
    # assamble agent's variables
    vars = prompt.get_variables()
    index = 0
    for agent in agents:
        personInfo = """You are a person and your basic information is as follows:
    {}{}{}{}""".format(agent.Brain.Working.Reason['genderDescription'], agent.Brain.Working.Reason['educationDescription'], agent.Brain.Working.Reason['consumptionDescription'], agent.Brain.Working.Reason['occupationDescription'])
        vars[index][0] = personInfo
        if 'lifestyle' in agent.Brain.Working.Reason.keys():
            lifestyle = agent.Brain.Working.Reason['lifestyle']
        else:
            lifestyle = "It depends on weekday or weekend. When weekday people generally go to work and do some other things between work. It is important to note that people generally do not work on weekends and prefer entertainment, sports and leisure activities. There will also be more freedom in the allocation of time."
        vars[index][1] = lifestyle
        if 'day' in agent.Brain.Working.Reason.keys():
            day = f"Today is {agent.Brain.Working.Reason['day']}"
        else:
            day = "Today is Monday."
        vars[index][2] = day
        index += 1
    prompt.set_variables(vars)

    # get prompt
    user_prompt = prompt.get_prompt()
    system_prompt = f"Just answer a number x that refers to a hour of in 24-hour, no reasons needed."
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]

    # request
    response = await agents[0].Soul.atext_request(messages)
    return response


async def generate_wake_up_hour(wk:WorkingMemory):
    personInfo = """You are a person and your basic information is as follows:
{}{}{}{}""".format(wk.Reason['genderDescription'], wk.Reason['educationDescription'], wk.Reason['consumptionDescription'], wk.Reason['occupationDescription'])
    lifestyle = "It depends on weekday or weekend. When weekday people generally go to work and do some other things between work. It is important to note that people generally do not work on weekends and prefer entertainment, sports and leisure activities. There will also be more freedom in the allocation of time."
    if 'day' in wk.Reason.keys():
        day = f"Today is {wk.Reason['day']}"
    else:
        day = "Today is Monday."
    
    prompt_template = "prompt_template/wake_up_hour_v1.txt"
    user_prompt = generate_prompt([personInfo, lifestyle, day], prompt_template)
    system_prompt = f"Just answer a number x that refers to a hour of in 24, no reasons needed."

    prompt = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]

    response = await wk._agent.Soul.atext_request(prompt)
    # print("generate the wake up hour: \n", response)
    if '0' == response[0]:
        response = response[1:]

    return int(eval(response))

# [ ] Batch Prompt
async def generate_first_daily_plan_batch(agents, prompt, wake_up_hours):
    # assamble agent's variables
    vars = prompt.get_variables()
    index = 0
    for agent in agents:
        personInfo = """You are a person and your basic information is as follows:
    {}{}{}{}""".format(agent.Brain.Working.Reason['genderDescription'], agent.Brain.Working.Reason['educationDescription'], agent.Brain.Working.Reason['consumptionDescription'], agent.Brain.Working.Reason['occupationDescription'])
        vars[index][0] = personInfo
        if 'lifestyle' in agent.Brain.Working.Reason.keys():
            lifestyle = agent.Brain.Working.Reason['lifestyle']
        else:
            lifestyle = "It depends on weekday or weekend. When weekday people generally go to work and do some other things between work. It is important to note that people generally do not work on weekends and prefer entertainment, sports and leisure activities. There will also be more freedom in the allocation of time."
        vars[index][1] = lifestyle
        if 'day' in agent.Brain.Working.Reason.keys():
            day = f"Today is {agent.Brain.Working.Reason['day']}"
        else:
            day = "Today is Monday."
        vars[index][2] = day
        vars[index][3] = wake_up_hours[index]
        index += 1
    prompt.set_variables(vars)

    # get prompt
    user_prompt = prompt.get_prompt()
    
    messages = [{"role":"user", "content": user_prompt}]

    response = await agents[0].Soul.atext_request(messages)
    # print("generate the first daily plan: \n", response)
    return response


async def generate_first_daily_plan(wk, wake_up_hour):
    personInfo =  personInfo = """You are a person and your basic information is as follows:
{}{}{}{}""".format(wk.Reason['genderDescription'], wk.Reason['educationDescription'], wk.Reason['consumptionDescription'], wk.Reason['occupationDescription'])
    if 'lifestyle' in wk.Reason.keys():
        lifestyle = wk.Reason['lifestyle']
    else:
        lifestyle = "It depends on weekday or weekend. When weekday people generally go to work and do some other things between work (like eating, shopping, sproting, etc.). It is important to note that people generally do not work on weekends and prefer entertainment, sports and leisure activities. There will also be more freedom in the allocation of time."
    if 'day' in wk.Reason.keys():
        day = f"{wk.Reason['day']}"
    else:
        day = "Monday."

    prompt_template = "prompt_template/daily_planning_v6.txt"
    user_prompt = generate_prompt([personInfo, lifestyle, day, wake_up_hour], prompt_template)
    
    prompt = [{"role":"user", "content": user_prompt}]

    response = await wk._agent.Soul.atext_request(prompt)
        # token_cost = 0
    
    # print("generate the first daily plan: \n", response)

    return f"1) {response}"


def generate_intention(wk, daily_plan, now_time):
    system_prompt = """You should generate your schudule for today, and you need to consider how your character attributes relate to your behavior and follow your daily plan.
Note that: 
1. All times are in 24-hour format.
2. The generated schedule must start at 0:00 and end at 24:00. Don't let your schedule spill over into the next day.
3. Must remember that events can only be choosed from [go to work, go home, eat, do shopping, do sports, excursion, leisure or entertainment, go to sleep, medical treatment, handle the trivialities of life, banking and financial services, cultural institutions and events].
Answer in the json format and keys are ["event_name", "reasons"]."""
    personInfo = """You are a person and your basic information is as follows:
{}{}{}{}""".format(wk.Reason['genderDescription'], wk.Reason['educationDescription'], wk.Reason['consumptionDescription'], wk.Reason['occupationDescription'])
    if 'lifestyle' in wk.Reason.keys():
        lifestyle = wk.Reason['lifestyle']
    else:
        lifestyle = "It depends on weekday or weekend. When weekday people generally go to work and do some other things between work (like eating, shopping, sproting, etc.). It is important to note that people generally do not work on weekends and prefer entertainment, sports and leisure activities. There will also be more freedom in the allocation of time."
    if 'day' in wk.Reason.keys():
        day = f"{wk.Reason['day']}"
    else:
        day = "Monday."
    
    prompt_template = "prompt_template/plan_to_hourly_schudule.txt"
    user_prompt = generate_prompt([personInfo, lifestyle, day, now_time, wk.Reason['history'], daily_plan], prompt_template)
    # print(user_prompt)

    prompt = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    response = wk._agent.Soul.text_request(prompt)
        # token_cost = 0
    # print("daily plan to hourly schdule : \n", response)

    return response

def generate_intention_time(wk, intention, now_time):
    system_prompt = """Estimate the time you spend on current arragement.
You must consider some fragmented time, such as 3 hours plus 47 minute, and 7 hours and 13 minutes. Answer in the json format and keys are ["hours", "minutes"]"""
    personInfo =  personInfo = """You are a person and your basic information is as follows:
{}{}{}{}""".format(wk.Reason['genderDescription'], wk.Reason['educationDescription'], wk.Reason['consumptionDescription'], wk.Reason['occupationDescription'])
    if 'lifestyle' in wk.Reason.keys():
        lifestyle = wk.Reason['lifestyle']
    else:
        lifestyle = "It depends on weekday or weekend. When weekday people generally go to work and do some other things between work (like eating, shopping, sproting, etc.). It is important to note that people generally do not work on weekends and prefer entertainment, sports and leisure activities. There will also be more freedom in the allocation of time."
    if 'day' in wk.Reason.keys():
        day = f"{wk.Reason['day']}"
    else:
        day = "Monday."

    prompt_template = "prompt_template/intention_time_estimation.txt"
    user_prompt = generate_prompt([personInfo, lifestyle, day, now_time, wk.Reason['history'], wk.Reason['daily_plan'], intention], prompt_template)

    prompt = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    response = wk._agent.Soul.text_request(prompt)
        # token_cost = 0

    # print("intention time estimation : \n", response)

    return response


async def generate_reflected_memory(wk, memory):
    user_prompt = f"""Here are your memory stream of today until now: {memory} 
You should summarize your memory into one sentence.
The summarized memory:
"""

    prompt = [{"role":"user", "content": user_prompt}]
    response = await wk._agent.Soul.atext_request(prompt)

    # print("reflected memory : \n", response)

    return response


def generate_retrieved_plan(wk, history, daily_plan, now_time):
    personInfo = """You are a person and your basic information is as follows:
{}{}{}{}""".format(wk.Reason['genderDescription'], wk.Reason['educationDescription'], wk.Reason['consumptionDescription'], wk.Reason['occupationDescription'])
    user_prompt = f"""{personInfo};
Your daily plans are as follows: {daily_plan};
The things you have done are {history};
And now is {now_time}, Which of the plans is best for execution now? Answer in json format, and keys are ['plan', 'reason'].
"""
    prompt = [{"role":"user", "content": user_prompt}]
    response = wk._agent.Soul.text_request(prompt)

    # print("retrieved plan : \n", response)

    return response

def generate_intention_projector(wk, intention):
    system_prompt = "You are role as a projector to project one itme into another item."
    user_prompt = f"""INPUT: {intention}
Candidate_answer: [go to work, go home, eat, do shopping, do sports, excursion, leisure or entertainment, go to sleep, medical treatment, handle the trivialities of life, banking and financial services, cultural institutions and events]
Query: Choose the most related one item in Candidate_answer;
Answer: 
"""
    prompt = [
        {"role":"system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]

    prompt = [{"role":"user", "content": user_prompt}]
    response = wk._agent.Soul.text_request(prompt)

    # print("intention projector : \n", response)

    return response

async def generate_action_from_planning(wk, perceived, daily_plan, now_time):
    personInfo = wk.Reason['personInfo']
    user_prompt = f"""{personInfo}; 
Your daily plan is {daily_plan}
The arrangement of the previous things you have done is as follows: {wk.Reason['memory']}.
Now time is {now_time}, and you have perceived these places you can go: {perceived};
What's the next arrangement just now? Which place will you go next, specifically the name? And how long will you stay there, with exactly hours and minutes, always more than 1 hour. (If you just stay here, you should also answer the current place and stay time.) Please output the only answer and explain your reasons for your choice. Answer in the json format and keys are ["reason", "arrangement", "next_place", "hours", "minutes"].
"""
    prompt = [{"role":"user", "content": user_prompt}]
    response = await wk._agent.Soul.atext_request(prompt)

    # print("=============================\n", user_prompt)
    # print("generate_action_from_planning : \n", response)

    return response