# Dir description
- **DataLocal**: used for sample agent profile from different cities
- **map_cache**: used for map file storage
    - please download map file from: **coming soon**
- **output**: the initial output dir
- **prompt_template**: prompt template used for LLM request
    - you can reuse the prompt we have generated or you can generate the prompt by you self
    - check **[opencity](./OpenCity.ipynb)**
- **prototype**: the initial prototype and agent profile storage dir

# File description
- **config.yaml**: this file is the basic config file, change the **api-key** and choice **model** for issuing LLM requests.
- **generativeAgent.py**: implementation of generative agent, single and group format
- **util_gpt**: util functions for generative agent
- **utils**: global utils
- **IPL.ipynb**: the in-context prototype learning
- **OpenCity**: main entrance for simulating generative agents in OpenCity
