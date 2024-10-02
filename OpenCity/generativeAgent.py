import yaml
import time
import json
import random
from utils import *
from util_gpt import *
import asyncio

import pycityagent
import yaml
from pycityagent.ac.action import *
from pycityagent.ac.hub_actions import *
from pycityagent.ac.sim_actions import *
from pycityagent import FuncAgent
from pycityagent.brain.memory import (LTMemory, MemoryPersistence,
                                      MemoryReason, MemoryRetrive)
from pycityagent.urbanllm import LLMConfig, UrbanLLM
from pycityagent.brain.static import POI_TYPE_DICT
from pycitysim import *

class FakeSimulator:
    """
    A fake simulator class
    Due to the need for bilateral blind review, some sensitive information exists in the original simulator class for data download and other services
    This is a simulator class contains only map data
    """
    def __init__(self) -> None:
        self.map = None
    
    async def GetFuncAgent(self, name:str) -> FuncAgent:
        agent = FuncAgent(
                    name,
                    "",
                    simulator=self
                )
        agent.set_streetview_config(self.config['streetview_request'])
        return agent


def spatial_node_construct(poi, relation):
    node = {}
    node["id"] = poi['id']
    node["name"] = poi['name']
    if poi["category"] in POI_TYPE_DICT.keys():
        node["category"] = POI_TYPE_DICT[poi['category']]
    else:
        node["category"] = poi['category']
    node["aoi_id"] = poi['aoi_id']
    node["relation"] = relation
    return node

class Spatial_Memory:
    def __init__(self) -> None:
        self.visited = []
        self.scene = []
        self.spatial_dict = {}

    def add_memory_from_poi(self, poi, relation, label="visited"):
        node = spatial_node_construct(poi, relation)
        self.add_memory_from_node(node, label)

    def update_node(self, node):
        pass

    def clear_scene(self):
        self.scene = []

    def add_scene(self, node_id):
        if node_id not in self.scene:
            self.scene.append(node_id)
    
    def add_visited(self, node_id):
        if node_id not in self.visited:
            self.visited.append(node_id)
    
    def add_memory_from_node(self, node, label="visited"):
        id = node['id']
        if id not in self.spatial_dict.keys():
            self.spatial_dict[node['id']] = node
        else:
            self.update_node(node)
        if label == "visited":
            self.add_visited(id)
        else:
            self.add_scene(id)

    def to_str(self, label="visited"):
        if label == "visited":
            out_dict = [self.spatial_dict[i] for i in self.visited]
        else:
            out_dict = [self.spatial_dict[i] for i in (self.visited+self.scene)]
        out_str = ""
        for count, node in enumerate(out_dict):
            out_str += f"Place {count+1} : {node}\n"
            count += 1
        return out_str

def add_memory(wk, labels):
    count = len(wk.Reason['memory'])
    if labels == "daily_plan":
        wk.Reason['memory'].append(f"({count}). {wk.Reason['now_time']}: you make the plan: {wk.Reason['daily_plan']};")
    elif labels == "intention":
        wk.Reason['memory'].append(f"({count}). {wk.Reason['now_time']}: you decide to {wk.Reason['intention']};")
    elif labels == 'location':
        wk.Reason['memory'].append(f"({count}). you go to {wk.Reason['nowPlace'][0]};")


async def reflect(wk):
    if len(wk.Reason['memory']) < 10:
        return 0
    memory = wk.Reason['memory'][:10]
    response = await generate_reflected_memory(wk, memory)

    wk.Reason['memory'] = [f"(0). {response}"] + [f"({i+1}). {m[m.find('.'):]}" for i, m in enumerate(wk.Reason['memory'][10:])]


async def perceive(wk):
    # print(wk._agent.motion)
    pois = wk._agent._simulator.map.query_pois(
        center = (wk._agent.motion['position']['xy_position']['x'], wk._agent.motion['position']['xy_position']['y']), 
        radius = 200,  
        category_prefix= "", 
        limit = 20      # 限制了perceive的POI数量，防止过多
    )  
    for poi, _ in pois:
        wk.Reason['spatial'].add_memory_from_poi(poi, "You can see it from here.", "scene")
    return wk.Reason['spatial'].to_str("all")

async def plan(wk):
    async def _long_term_planning(wk, new_day):
        if new_day == "New Day":
            wake_up_hour = await generate_wake_up_hour(wk)
            increment_minutes = wake_up_hour * 60
            end_time, _ = add_time(wk.Reason['now_time'], increment_minutes)
            intent = "Sleep at home"
            set_time = "("+wk.Reason['now_time']+", "+end_time+")"
            thisThing = [intent, set_time, wk.Reason['Homeplace']]
            count = len(wk.Reason['memory'])
            wk.Reason['memory'].append(f"({count}). {wk.Reason['now_time']}: you decide to sleep at {wk.Reason['Homeplace'][0]};")
            wk.Reason['trajectory'].append(thisThing)
            wk.Reason['history'].append([intent, set_time])
            wk.Reason['now_time'] = end_time
            daily_plan = await generate_first_daily_plan(wk, wake_up_hour)
        wk.Reason['daily_plan'] = daily_plan
        return daily_plan

    async def _plan2action(wk):
        def __checkPlace(loc_str, nxt):
            loc_list = loc_str.split('\n')
            for loc in loc_list:
                if nxt in loc:
                    p = json.loads(loc[loc.find(':')+1:].replace('\'','\"'))
                    return p
            return None

        perceived = wk.Reason['perceived']
        daily_plan = wk.Reason['daily_plan']
        now_time = wk.Reason['now_time']
        genOK = False
        try_count = 5
        while not genOK:
            try:
                response = await generate_action_from_planning(wk, perceived, daily_plan, now_time)

                if "```json" in response:
                    response = json.loads(response.split('```json')[1].split('```')[0])
                else:
                    response = json.loads(response)
                arrangement = response['arrangement']
                nxt_loc = response['next_place']
                nxt_poi = __checkPlace(perceived, nxt_loc)
                hours = int(response['hours'])
                minutes = int(response['minutes'])
                if nxt_poi is not None: genOK = True  
                else: try_count -= 1
            except Exception as err:
                print(err)
                try_count -= 1
                if try_count < 0: raise Exception("Error when using LLM")
        
        increment_minutes = 60*hours+minutes
        noiseTime = sampleNoiseTime()
        if increment_minutes + noiseTime > 0:  # 防止变成负数
            increment_minutes = increment_minutes + noiseTime  # 转化成分钟数量
        
        end_time, cross_day = add_time(now_time, increment_minutes)
        return arrangement, nxt_poi, end_time, cross_day

    new_day = wk.Reason['new_day']
    if new_day:
        wk.Reason['daily_plan'] = await _long_term_planning(wk, new_day)
        add_memory(wk, 'daily_plan')

    daily_plan = wk.Reason['daily_plan']
    now_time = wk.Reason['now_time']
    history = wk.Reason['history']
    
    current_intention, nxt_poi, end_time, cross_day = await _plan2action(wk)

    wk.Reason['spatial'].add_memory_from_node(nxt_poi, f"You went here for {current_intention} from {now_time} to {end_time}.")

    if cross_day or end_time == "23:59":
        seTime = "("+ now_time+", 23:59)"
        history.append([current_intention, seTime])
        wk.Reason['break'] = True
    else:
        seTime = "("+ now_time+", "+end_time+")"
        history.append([current_intention, seTime])
        
        # print(history)
        gapTime = sampleGapTime()
        tmpnowTime, cross_day = add_time(end_time, gapTime)  
        if cross_day:
            now_time = end_time
        else:
            now_time = tmpnowTime
        wk.Reason['break'] = False

    wk.Reason['intention'] = current_intention
    add_memory(wk, "intention")
    wk.Reason['now_time'] = now_time
    # print([current_intention, seTime])
    wk.Reason['history'] = history
    trajectory = wk.Reason['trajectory']
    # print('before query-nowPlace: {}'.format(wk.Reason['nowPlace']))
    nextPlace = (nxt_poi['name'], nxt_poi['id'])
    intent, setime = wk.Reason['history'][-1]
    thisThing = [intent, setime, nextPlace]
    trajectory.append(thisThing)
    wk.Reason['nowPlace'] = nextPlace
    add_memory(wk, 'location')
    # print('after query-nowPlace: {}\n'.format(wk.Reason['nowPlace']))
    
    return trajectory

class GAGroup:
    def __init__(self, config_file_path:str, agent_prefix:str, city:str, hub:bool=False) -> None:
        with open(config_file_path, 'r') as file:
            self.config = yaml.safe_load(file)
        self.city = city
        self.hub = hub
        print("---Init simulator and map")
        self.sim = FakeSimulator()
        if city == 'beijing':
            self.sim.map = map.Map(
                mongo_uri = "",
                mongo_db = "llmsim",
                mongo_coll = "map_beijing5ring_withpoi_0424",
                cache_dir = "./map_cache",
            )
            cut_map_range(self.sim.map, nw=[116.115, 40.17], se=[116.695, 39.7])
        elif city == 'new_york':
            self.sim.map = map.Map(
                mongo_uri = "",
                mongo_db = "llmsim",
                mongo_coll = "map_newyork_20240808",
                cache_dir = "./map_cache",
            )
            poly = get_polygon_from_file('../cache/new_york.geojson')
            cut_map_poly(self.sim.map, poly)
        elif city == 'san_francisco':
            self.sim.map = map.Map(
                mongo_uri = "",
                mongo_db = "llmsim",
                mongo_coll = "map_san_francisco_20240807",
                cache_dir = "./map_cache",
            )
            poly = get_polygon_from_file('./map_cache/san_francisco.geojson')
            cut_map_poly(self.sim.map, poly)
        elif city == 'london':
            self.sim.map = map.Map(
                mongo_uri = "",
                mongo_db = "llmsim",
                mongo_coll = "map_london_20240807",
                cache_dir = "./map_cache",
            )
            cut_map_range(self.sim.map, nw=[-0.315578, 51.614099], se=[0.107305, 51.3598])
        elif city == 'paris':
            self.sim.map = map.Map(
                mongo_uri = "",
                mongo_db = "llmsim",
                mongo_coll = "map_paris_20240808",
                cache_dir = "./map_cache",
            )
            # [(48.870079, 2.371342), (48.850375, 2.392858)]
            cut_map_range(self.sim.map, nw=[2.24924, 48.903659], se=[2.423433, 48.812177])
        elif city == 'sydney':
            self.sim.map = map.Map(
                mongo_uri = "",
                mongo_db = "llmsim",
                mongo_coll = "map_sydney_20240807",
                cache_dir = "./map_cache",
            )
            # [(-33.874652, 151.217697), (-33.889818, 151.240466)]
            cut_map_range(self.sim.map, nw=[150.839882, -33.721288], se=[151.243988, -34.0432])
        else:
            raise Exception("Wrong City")
        print("---Create LLM client")
        self.llm_config = LLMConfig(self.config['llm_request'])
        self.soul = UrbanLLM(self.llm_config)
        self.prefix = agent_prefix
        self.agents = []
        self.reasons = []
        self.reasons.append(MemoryReason("perceived", perceive, "stage 0: perceive around pois"))
        self.reasons.append(MemoryReason("trajectory", plan, "stage1: "))
        self.reasons.append(MemoryReason("reflect", reflect, "stage3: reflect the memory"))

    def clear_agents(self):
        print("Clearing Agents...")
        self.agents = []

    async def prepare_agents(self, noa:int, day:str, from_storage:str=None):
        self.noa = noa

        if day not in ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']:
            print("Wrong day, only support ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']")
            return
        
        self.clear_agents()
        if from_storage != None:
            with open(from_storage, 'r') as f:
                meta_data = json.load(f)
            homes = []
            works = []
            educations = []
            genders = []
            consumptions = []
            occupations = []
            self.agent_groups = []
            self.ids = []
            for item in meta_data:
                homes.append(item['home'])
                works.append(item['work'])
                educations.append(item['educationDescription'])
                genders.append(item['genderDescription'])
                consumptions.append(item['consumptionDescription'])
                occupations.append(item['occupationDescription'])
                if 'group' in item.keys():
                    self.agent_groups.append(item['group'])
                if 'id' in item.keys():
                    self.ids.append(item['id'])
        else:
            (homes, works, cbg_ids) = choiceHW(self.city, self.sim.map, number=noa)
            educations, genders, consumptions, occupations = choiceProfile(self.city, cbg_ids, number=noa)
        if len(self.agents) != self.noa:
            print(f"Preparing {self.noa} Agents...")
            for i in range(self.noa):
                agent = await self.sim.GetFuncAgent(self.prefix+f"_{i}")
                agent.StateTransformer.reset_state_transformer(['Generative'], 'Generative')
                agent.StateTransformer.add_transition(trigger='start', source='*', dest='Generative')
                agent.Brain.Memory.Working.reset_reason()
                agent.add_soul(self.soul)
                if self.hub:
                    agent.ConnectToHub(self.config['apphub_request'])
                    agent.Bind()

                home = homes[i]
                work = works[i]
                home_name = self.sim.map.get_poi(home[1])['name']
                try:
                    work_name = self.sim.map.get_poi(work[1])['name']
                except:
                    # not in range:
                    work_name = "Unknown"
                # 0是POI的name,1是POI的id,而且是int类型的
                Homeplace = (home_name, home[1])
                Workplace = (work_name, work[1])

                if from_storage != None:
                    education = educations[i]
                    gender = genders[i]
                    consumption = consumptions[i]
                    occupation = occupations[i]

                    genderDescription = "" if gender == 'uncertain' else "Gender: {}; ".format(gender)
                    educationDescription = "" if education == 'uncertain' else "Education: {}; ".format(education)
                    consumptionDescription = "" if consumption == 'uncertain' else "Consumption level: {}; ".format(consumption)
                    occupationDescription = "" if occupation == 'uncertain' else "Occupation or Job Content: {};".format(occupation)
                else:
                    genderDescription = genders[i]
                    educationDescription = educations[i]
                    consumptionDescription = consumptions[i]
                    occupationDescription = occupations[i]

                agent.Brain.Memory.Working.Reason = {}
                agent.Brain.Memory.Working.Reason['genderDescription'] = genderDescription
                agent.Brain.Memory.Working.Reason['educationDescription'] = educationDescription
                agent.Brain.Memory.Working.Reason['consumptionDescription'] = consumptionDescription
                agent.Brain.Memory.Working.Reason['occupationDescription'] = occupationDescription
                agent.Brain.Memory.Working.Reason['day'] = day
                agent.Brain.Memory.Working.Reason['intentIndex'] = 0
                agent.Brain.Memory.Working.Reason['history'] = []
                agent.Brain.Memory.Working.Reason['memory'] = []
                agent.Brain.Memory.Working.Reason['now_time'] = '00:00'
                agent.Brain.Memory.Working.Reason['trajectory'] = []
                agent.Brain.Memory.Working.Reason['new_day'] = 'New Day'
                agent.Brain.Memory.Working.Reason['break'] = False
                agent.Brain.Memory.Working.Reason['start_time'] = time.time()
                agent.Brain.Memory.Working.Reason['spatial'] = Spatial_Memory()
                agent.Brain.Memory.Working.Reason['personInfo'] = """You are a person and your basic information is as follows:
    {}{}{}{}""".format(genderDescription, educationDescription, consumptionDescription, occupationDescription)
                
                agent.Brain.Memory.Working.Reason['spatial'].add_memory_from_poi(self.sim.map.get_poi(home[1]), "Your Home.")
                try:
                    agent.Brain.Memory.Working.Reason['spatial'].add_memory_from_poi(self.sim.map.get_poi(work[1]), "Your Working Place.")
                except:
                    pass
                nowPlace = Homeplace  
                await agent.set_position_poi(nowPlace[1], hub=self.hub)
                agent.Brain.Memory.Working.Reason['nowPlace'] = nowPlace  # 名称和地点
                agent.Brain.Memory.Working.Reason['Homeplace'] = Homeplace  # 名称和地点
                agent.Brain.Memory.Working.Reason['Workplace'] = Workplace  # 名称和地点
                agent.Brain.Memory.Working.add_reason('Generative', self.reasons[0])
                agent.Brain.Memory.Working.add_reason('Generative', self.reasons[1])
                agent.Brain.Memory.Working.add_reason('Generative', self.reasons[2])
        
                self.agents.append(agent)
    
    async def srun(self, id):
        while True:
            self.agents[id].Brain.Memory.Working.Reason['spatial'].clear_scene()
            await self.agents[id].Brain.Memory.Working.runReason()
            self.agents[id].Brain.Memory.Working.Reason['new_day'] = None
            await self.agents[id].set_position_poi(self.agents[id].Brain.Memory.Working.Reason['nowPlace'][1], hub=self.hub)
            
            if self.agents[id].Brain.Memory.Working.Reason['break']:
                break
        print(f"Agent_{id} Finished...")

    async def run(self):
        print("Start simulation...")
        start = time.time()
        tasks = [self.srun(i) for i in range(self.noa)]
        await asyncio.gather(*tasks)
        end = time.time()
        print("Finished simulation...")
        print(f"Time for {self.noa} Agents: {end-start:.2f} s")


class GenerativeAgent:
    def __init__(self, config_file_path:str, agent_id:int, agent_name:str, city:str, hub:bool=False) -> None:
        with open(config_file_path, 'r') as file:
            self.config = yaml.safe_load(file)
        self.city = city
        self.hub = hub
        print("---Init simulator and map")
        self.sim = FakeSimulator()
        if city == 'beijing':
            self.sim.map = map.Map(
                mongo_uri = "",
                mongo_db = "llmsim",
                mongo_coll = "map_beijing5ring_withpoi_0424",
                cache_dir = "./map_cache",
            )
            cut_map_range(self.sim.map, nw=[116.115, 40.17], se=[116.695, 39.7])
        elif city == 'new_york':
            self.sim.map = map.Map(
                mongo_uri = "",
                mongo_db = "llmsim",
                mongo_coll = "map_newyork_20240808",
                cache_dir = "./map_cache",
            )
            poly = get_polygon_from_file('../cache/new_york.geojson')
            cut_map_poly(self.sim.map, poly)
        elif city == 'san_francisco':
            self.sim.map = map.Map(
                mongo_uri = "",
                mongo_db = "llmsim",
                mongo_coll = "map_san_francisco_20240807",
                cache_dir = "./map_cache",
            )
            poly = get_polygon_from_file('./map_cache/san_francisco.geojson')
            cut_map_poly(self.sim.map, poly)
        elif city == 'london':
            self.sim.map = map.Map(
                mongo_uri = "",
                mongo_db = "llmsim",
                mongo_coll = "map_london_20240807",
                cache_dir = "./map_cache",
            )
            cut_map_range(self.sim.map, nw=[-0.315578, 51.614099], se=[0.107305, 51.3598])
        elif city == 'paris':
            self.sim.map = map.Map(
                mongo_uri = "",
                mongo_db = "llmsim",
                mongo_coll = "map_paris_20240808",
                cache_dir = "./map_cache",
            )
            # [(48.870079, 2.371342), (48.850375, 2.392858)]
            cut_map_range(self.sim.map, nw=[2.24924, 48.903659], se=[2.423433, 48.812177])
        elif city == 'sydney':
            self.sim.map = map.Map(
                mongo_uri = "",
                mongo_db = "llmsim",
                mongo_coll = "map_sydney_20240807",
                cache_dir = "./map_cache",
            )
            # [(-33.874652, 151.217697), (-33.889818, 151.240466)]
            cut_map_range(self.sim.map, nw=[150.839882, -33.721288], se=[151.243988, -34.0432])
        else:
            raise Exception("Wrong City")
        print("---Create LLM client")
        self.llm_config = LLMConfig(self.config['llm_request'])
        self.soul = UrbanLLM(self.llm_config)
        self.reasons = []
        self.agent_id = agent_id
        self.agent_name = agent_name
        self.agent = None
        self.reasons.append(MemoryReason("perceived", perceive, "stage 0: perceive around pois"))
        self.reasons.append(MemoryReason("trajectory", plan, "stage1: "))
        self.reasons.append(MemoryReason("reflect", reflect, "stage3: reflect the memory"))

    async def run(self, day:str):
        if day not in ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']:
            print("Wrong day, only support: ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']")
            return
        
        if self.agent == None:
            self.agent = await self.sim.GetFuncAgent(self.agent_name)
            self.agent.StateTransformer.reset_state_transformer(['Generative'], 'Generative')
            self.agent.StateTransformer.add_transition(trigger='start', source='*', dest='Generative')
        if self.hub:
            self.agent.ConnectToHub(self.config['apphub_request'])
            self.agent.Bind()

        (home, work, cbg_id) = choiceHW(self.city, self.sim.map)
        # 0是POI的name,1是POI的id,而且是int类型的
        home_name = self.sim.map.get_poi(home[0][1])['name']
        try:
            work_name = self.sim.map.get_poi(work[0][1])['name']
        except:
            # not in range:
            work_name = "Un-Known"
        Homeplace = (home_name, home[0][1])
        Workplace = (work_name, work[0][1])

        # randomly load profiles
        education, gender, consumption, occupation = choiceProfile(self.city, cbg_id)
        self.genderDescription = "" if gender == 'uncertain' else "Gender: {}; ".format(gender[0])
        self.educationDescription = "" if education == 'uncertain' else "Education: {}; ".format(education[0])
        self.consumptionDescription = "" if consumption == 'uncertain' else "Consumption level: {}; ".format(consumption[0])
        self.occupationDescription = "" if occupation == 'uncertain' else "Occupation or Job Content: {};".format(occupation[0])

        self.agent.add_soul(self.soul)
        self.agent.Brain.Memory.Working.reset_reason()
        self.agent.Brain.Memory.Working.Reason = {}
        self.agent.Brain.Memory.Working.Reason['genderDescription'] = self.genderDescription
        self.agent.Brain.Memory.Working.Reason['educationDescription'] = self.educationDescription
        self.agent.Brain.Memory.Working.Reason['consumptionDescription'] = self.consumptionDescription
        self.agent.Brain.Memory.Working.Reason['occupationDescription'] = self.occupationDescription
        self.agent.Brain.Memory.Working.Reason['day'] = day
        self.agent.Brain.Memory.Working.Reason['intentIndex'] = 0
        self.agent.Brain.Memory.Working.Reason['history'] = []
        self.agent.Brain.Memory.Working.Reason['memory'] = []
        self.agent.Brain.Memory.Working.Reason['now_time'] = '00:00'
        self.agent.Brain.Memory.Working.Reason['trajectory'] = []
        self.agent.Brain.Memory.Working.Reason['new_day'] = 'New Day'
        self.agent.Brain.Memory.Working.Reason['start_time'] = time.time()
        self.agent.Brain.Memory.Working.Reason['spatial'] = Spatial_Memory()
        self.agent.Brain.Memory.Working.Reason['personInfo'] = """You are a person and your basic information is as follows:
{}{}{}{}""".format(self.genderDescription, self.educationDescription, self.consumptionDescription, self.occupationDescription)


        self.agent.Brain.Memory.Working.Reason['spatial'].add_memory_from_poi(self.sim.map.get_poi(home[0][1]), "Your Home.")
        try:
            self.agent.Brain.Memory.Working.Reason['spatial'].add_memory_from_poi(self.sim.map.get_poi(work[0][1]), "Your Working Place.")
        except:
            pass
        nowPlace = Homeplace  
        await self.agent.set_position_poi(nowPlace[1], hub=self.hub)

        self.agent.Brain.Memory.Working.Reason['nowPlace'] = nowPlace 
        self.agent.Brain.Memory.Working.Reason['Homeplace'] = Homeplace 
        self.agent.Brain.Memory.Working.Reason['Workplace'] = Workplace 
        
        self.agent.Brain.Memory.Working.add_reason('Generative', self.reasons[0])
        self.agent.Brain.Memory.Working.add_reason('Generative', self.reasons[1])
        self.agent.Brain.Memory.Working.add_reason('Generative', self.reasons[2])

        while True:
            self.agent.Brain.Memory.Working.Reason['spatial'].clear_scene()
            await self.agent.Brain.Memory.Working.runReason()
            self.agent.Brain.Memory.Working.Reason['new_day'] = None
            await self.agent.set_position_poi(self.agent.Brain.Memory.Working.Reason['nowPlace'][1], hub=self.hub)
            if self.agent.Brain.Memory.Working.Reason['break']: 
                break

if __name__ == "__main__":
    a = GenerativeAgent(0, "a")
    a.print_prompts()